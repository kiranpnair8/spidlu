from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spidlu.config import load_config
from spidlu.eval import causal_lm_nll_from_logits, compare_hf_loss, downstream_accuracy, shifted_causal_targets
from spidlu.layers import QuantizedActivationSTE, SpiDLU
from spidlu.metrics import count_parameters
from spidlu.phase1 import TRAINED_VARIANTS, build_run_context, state_fingerprint
from spidlu.surgery import Variant, apply_activation_surgery
from spidlu.train import train_variant


class FakeGatedMLP(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, hidden)
        self.up_proj = nn.Linear(hidden, hidden)
        self.down_proj = nn.Linear(hidden, hidden)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeBlock(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.mlp = FakeGatedMLP(hidden)

    def forward(self, x):
        return self.mlp(x)


class FakeCausalLM(nn.Module):
    base_model_prefix = "model"

    def __init__(self, vocab=17, hidden=8, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeBlock(hidden) for _ in range(layers)])
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False):
        x = self.embedding(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss, past_key_values=None)

    def generate(self, input_ids, max_new_tokens=2, **kwargs):
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self(generated).logits
            next_token = logits[:, -1:].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class StaticLogitModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("stored_logits", logits)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        logits = self.stored_logits[: input_ids.size(0), : input_ids.size(1)].clone()
        loss = None
        if labels is not None:
            targets = labels[:, 1:].clone()
            if attention_mask is not None:
                targets = targets.masked_fill(~attention_mask[:, 1:].bool(), -100)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                targets.contiguous().view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


def cfg():
    return SimpleNamespace(
        spidlu_alpha=0.9,
        spidlu_threshold=1.0,
        spidlu_T=4,
        quantized_levels=None,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_train_steps=1,
        max_train_tokens=None,
    )


def tiny_loader():
    input_ids = torch.randint(0, 17, (4, 6))
    return DataLoader([{"input_ids": row, "labels": row} for row in input_ids], batch_size=1)


def persistence_cfg(tmp_path, variant="spidlu", seed=42, smoke=True):
    return SimpleNamespace(
        output_dir=str(tmp_path),
        variants=[variant],
        seed=seed,
        smoke=smoke,
    )


def test_custom_causal_lm_nll_agrees_with_model_loss():
    logits = torch.zeros(1, 4, 8)
    logits[0, 0, 2] = 5.0
    logits[0, 1, 3] = 5.0
    logits[0, 2, 4] = 5.0
    model = StaticLogitModel(logits)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = input_ids.clone()
    comparison = compare_hf_loss(model, input_ids=input_ids, labels=labels)
    assert comparison["valid_tokens"].item() == 3
    assert torch.allclose(comparison["custom_loss"], comparison["hf_loss"])


def test_tiny_known_sequence_uses_shifted_labels_once():
    labels = torch.tensor([[10, 11, 12, 13]])
    targets, valid = shifted_causal_targets(labels)
    assert targets.tolist() == [[11, 12, 13]]
    assert valid.tolist() == [[True, True, True]]


def test_padding_does_not_contribute_to_nll():
    logits = torch.zeros(1, 4, 6)
    labels = torch.tensor([[1, 2, -100, -100]])
    attention_mask = torch.tensor([[1, 1, 0, 0]])
    nll, valid = causal_lm_nll_from_logits(logits, labels, attention_mask=attention_mask)
    assert valid.item() == 1
    assert torch.allclose(nll, torch.log(torch.tensor(6.0)))


def test_downstream_metric_matches_hand_computed_token_examples():
    logits = torch.zeros(2, 4, 6)
    logits[0, 0, 2] = 5.0
    logits[0, 1, 3] = 5.0
    logits[0, 2, 0] = 5.0
    logits[1, 0, 5] = 5.0
    logits[1, 1, 4] = 5.0
    logits[1, 2, 3] = 5.0
    model = StaticLogitModel(logits)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [2, 5, 4, 3]]),
        "labels": torch.tensor([[1, 2, 3, 4], [2, 5, 4, -100]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    }
    result = downstream_accuracy(model, [batch], torch.device("cpu"))
    assert result["downstream_metric"] == "next_token_token_accuracy"
    assert result["downstream_tokens"] == 5
    assert result["downstream_accuracy"] == 4 / 5
    assert result["downstream_sequence_exact_accuracy"] == 0.5
    assert result["downstream_chance_accuracy"] == 1 / 6


def test_feasibility_config_uses_shared_seed_and_all_variants():
    config = load_config(Path("configs") / "phase1_rq1_feasibility.yaml")
    assert config.seed == 42
    assert config.smoke is False
    assert config.variants == [
        "ann_original",
        "spidlu",
        "ann_compute_matched",
        "quantized_activation",
    ]
    assert config.dataset_name == "Salesforce/wikitext"
    assert config.dataset_config == "wikitext-2-raw-v1"
    assert config.eval_split == "validation"
    assert config.downstream_split == "validation"


def test_feasibility_trained_variants_share_reduced_budget():
    config = load_config(Path("configs") / "phase1_rq1_feasibility.yaml")
    assert config.max_train_steps == 8
    assert config.max_train_tokens == 4096
    assert config.save_every_steps == 4
    assert {variant.value for variant in TRAINED_VARIANTS} == {
        "spidlu",
        "ann_compute_matched",
        "quantized_activation",
    }
    assert "ann_original" not in {variant.value for variant in TRAINED_VARIANTS}


def test_spidlu_replaces_gated_activation_location():
    model = FakeCausalLM()
    records = apply_activation_surgery(model, Variant.SPIDLU, cfg())
    assert records
    assert isinstance(model.model.layers[0].mlp.act_fn, SpiDLU)
    assert records[0].semantic_location == "down_proj(act_fn(gate_proj(x)) * up_proj(x))"


def test_phase1_run_dirs_do_not_collide_for_variants(tmp_path):
    first = build_run_context(persistence_cfg(tmp_path, variant="spidlu"), run_id="same-run")
    second = build_run_context(
        persistence_cfg(tmp_path, variant="quantized_activation"),
        run_id="same-run",
    )
    assert first["run_dir"] != second["run_dir"]


def test_phase1_run_dirs_do_not_collide_for_seeds(tmp_path):
    first = build_run_context(persistence_cfg(tmp_path, seed=1), run_id="same-run")
    second = build_run_context(persistence_cfg(tmp_path, seed=2), run_id="same-run")
    assert first["run_dir"] != second["run_dir"]


def test_phase1_run_dir_requires_overwrite_for_same_run(tmp_path):
    cfg = persistence_cfg(tmp_path, variant="spidlu", seed=42)
    first = build_run_context(cfg, run_id="repeatable")
    try:
        build_run_context(cfg, run_id="repeatable")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected duplicate run_id to require overwrite.")
    second = build_run_context(cfg, run_id="repeatable", overwrite=True)
    assert first["run_dir"] == second["run_dir"]


def test_spidlu_forward_handles_changing_sequence_lengths():
    layer = SpiDLU(T=2)
    first = layer(torch.randn(1, 512, 8))
    second = layer(torch.randn(1, 7, 8))
    assert first.shape == (1, 512, 8)
    assert second.shape == (1, 7, 8)


def test_quantized_control_replaces_same_activation_location():
    model = FakeCausalLM()
    records = apply_activation_surgery(model, Variant.QUANTIZED_ACTIVATION, cfg())
    assert records
    assert isinstance(model.model.layers[0].mlp.act_fn, QuantizedActivationSTE)
    assert records[0].semantic_location == "down_proj(act_fn(gate_proj(x)) * up_proj(x))"


def test_ann_original_remains_structurally_unchanged():
    model = FakeCausalLM()
    original = type(model.model.layers[0].mlp.act_fn)
    records = apply_activation_surgery(model, Variant.ANN_ORIGINAL, cfg())
    assert records == []
    assert type(model.model.layers[0].mlp.act_fn) is original


def test_variants_start_from_identical_base_weights():
    fingerprints = []
    for _ in [
        Variant.ANN_ORIGINAL,
        Variant.SPIDLU,
        Variant.ANN_COMPUTE_MATCHED,
        Variant.QUANTIZED_ACTIVATION,
    ]:
        torch.manual_seed(123)
        fingerprints.append(state_fingerprint(FakeCausalLM()))
    assert len(set(fingerprints)) == 1


def test_parameter_counts_are_recorded_correctly():
    model = FakeCausalLM()
    counts = count_parameters(model)
    assert counts["parameters"] == sum(p.numel() for p in model.parameters())
    assert counts["trainable_parameters"] == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_all_four_variants_complete_smoke_mode():
    for variant in [
        Variant.ANN_ORIGINAL,
        Variant.SPIDLU,
        Variant.ANN_COMPUTE_MATCHED,
        Variant.QUANTIZED_ACTIVATION,
    ]:
        model = FakeCausalLM()
        apply_activation_surgery(model, variant, cfg())
        if variant != Variant.ANN_ORIGINAL:
            stats = train_variant(model, tiny_loader(), cfg(), torch.device("cpu"))
            assert stats["optimizer_steps"] == 1
        out = model(input_ids=torch.randint(0, 17, (1, 6)), labels=torch.randint(0, 17, (1, 6)))
        assert out.logits.shape[:2] == (1, 6)
