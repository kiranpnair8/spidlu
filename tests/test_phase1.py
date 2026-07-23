from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.aggregate_phase1 import paired_comparisons, publication_table, summarize
from spidlu.config import load_config
from spidlu.eval import causal_lm_nll_from_logits, compare_hf_loss, downstream_accuracy, shifted_causal_targets
from spidlu.layers import BlendedActivation, QuantizedActivationSTE, SpiDLU
from spidlu.metrics import count_parameters
from spidlu.phase1 import TRAINED_VARIANTS, build_run_context, state_fingerprint
from spidlu.surgery import (
    Variant,
    apply_activation_surgery,
    freeze_pretrained_for_activation_only,
    trainable_parameter_names,
)
from spidlu.train import (
    changed_trainable_parameters,
    clamp_trainable_blend_alphas,
    gradient_norms,
    optimizer_parameter_group_summary,
    train_variant,
    trainable_parameter_snapshot,
)


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
        save_every_steps=1,
        spidlu_alpha_max=0.1,
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


def test_corrected_feasibility_uses_function_preserving_low_alpha():
    config = load_config(Path("configs") / "phase1_rq1_corrected_feasibility.yaml")
    assert config.output_dir == "models/phase1_rq1_activation_feasibility"
    assert config.spidlu_function_preserving is True
    assert config.spidlu_alpha_mode == "trainable"
    assert config.spidlu_alpha_max == 0.1
    assert config.spidlu_warmup_steps == 8
    assert config.max_train_steps == 8


def test_phase1_aggregation_computes_mean_std():
    rows = [
        {"variant": "ann_original", "perplexity": 10.0, "downstream_accuracy": 0.5},
        {"variant": "ann_original", "perplexity": 14.0, "downstream_accuracy": 0.7},
        {"variant": "spidlu", "perplexity": 12.0, "downstream_accuracy": 0.6},
    ]
    summary = summarize(rows)
    by_key = {(row["variant"], row["metric"]): row for row in summary}
    assert by_key[("ann_original", "perplexity")]["mean"] == 12.0
    assert by_key[("ann_original", "perplexity")]["n"] == 2
    assert by_key[("spidlu", "downstream_accuracy")]["std"] == 0.0


def test_phase1_aggregation_computes_paired_comparisons():
    rows = [
        {"variant": "ann_original", "seed": 1, "perplexity": 10.0},
        {"variant": "ann_original", "seed": 2, "perplexity": 12.0},
        {"variant": "spidlu", "seed": 1, "perplexity": 11.0},
        {"variant": "spidlu", "seed": 2, "perplexity": 15.0},
    ]
    comparisons = paired_comparisons(rows, baselines=("ann_original",))
    row = next(item for item in comparisons if item["variant"] == "spidlu" and item["metric"] == "perplexity")
    assert row["n"] == 2
    assert row["mean_difference"] == 2.0
    assert row["percent_change_mean"] == 2.0 / 11.0 * 100.0
    assert row["p_value_method"] in {"paired_t_test_scipy", "paired_t_test_normal_approx"}


def test_phase1_publication_table_contains_p_value_column():
    summary = summarize([
        {"variant": "ann_original", "seed": 1, "perplexity": 10.0},
        {"variant": "ann_original", "seed": 2, "perplexity": 12.0},
        {"variant": "spidlu", "seed": 1, "perplexity": 11.0},
        {"variant": "spidlu", "seed": 2, "perplexity": 15.0},
    ])
    comparisons = paired_comparisons([
        {"variant": "ann_original", "seed": 1, "perplexity": 10.0},
        {"variant": "ann_original", "seed": 2, "perplexity": 12.0},
        {"variant": "spidlu", "seed": 1, "perplexity": 11.0},
        {"variant": "spidlu", "seed": 2, "perplexity": 15.0},
    ], baselines=("ann_original",))
    table = publication_table(summary, comparisons)
    assert "p-value" in table
    assert "spidlu" in table


def test_publication_config_uses_full_multiseed_budget_defaults():
    config = load_config(Path("configs") / "phase1_rq1_publication.yaml")
    assert config.smoke is False
    assert config.output_dir == "models/phase1_rq1_publication"
    assert config.max_train_steps == 100
    assert config.spidlu_alpha_mode == "trainable"
    assert config.spidlu_warmup_steps == 100
    assert config.spidlu_alpha_max == 0.1
    assert config.variants == [
        "ann_original",
        "spidlu",
        "ann_compute_matched",
        "quantized_activation",
    ]


def test_spidlu_replaces_gated_activation_location():
    model = FakeCausalLM()
    records = apply_activation_surgery(model, Variant.SPIDLU, cfg())
    assert records
    assert isinstance(model.model.layers[0].mlp.act_fn, BlendedActivation)
    assert isinstance(model.model.layers[0].mlp.act_fn.replacement_activation, SpiDLU)
    assert records[0].semantic_location == "down_proj(act_fn(gate_proj(x)) * up_proj(x))"


def test_function_preserving_spidlu_alpha_zero_matches_silu():
    x = torch.linspace(-4, 4, steps=17).reshape(1, 17)
    blended = BlendedActivation(
        nn.SiLU(),
        SpiDLU(alpha=0.9, threshold=1.0, T=4),
        blend_alpha=0.0,
        trainable=False,
    )
    assert torch.allclose(blended(x), nn.SiLU()(x), atol=1e-7, rtol=1e-7)


def test_ann_original_and_compute_matched_match_before_training():
    torch.manual_seed(123)
    original = FakeCausalLM()
    torch.manual_seed(123)
    compute_matched = FakeCausalLM()
    assert apply_activation_surgery(original, Variant.ANN_ORIGINAL, cfg()) == []
    assert apply_activation_surgery(compute_matched, Variant.ANN_COMPUTE_MATCHED, cfg()) == []
    input_ids = torch.randint(0, 17, (2, 6))
    with torch.inference_mode():
        original_logits = original(input_ids).logits
        matched_logits = compute_matched(input_ids).logits
    assert torch.allclose(original_logits, matched_logits)


def test_one_layer_surgery_changes_only_requested_module():
    model = FakeCausalLM(layers=4)
    scoped_cfg = SimpleNamespace(
        **cfg().__dict__,
        surgery_scope="one",
        surgery_layer_index=2,
        surgery_first_n=None,
    )
    records = apply_activation_surgery(model, Variant.SPIDLU, scoped_cfg)
    assert [record.layer_index for record in records] == [2]
    assert isinstance(model.model.layers[2].mlp.act_fn, BlendedActivation)
    assert all(isinstance(model.model.layers[idx].mlp.act_fn, nn.SiLU) for idx in (0, 1, 3))


def test_activation_only_variants_freeze_pretrained_weights():
    model = FakeCausalLM(layers=2)
    freeze_pretrained_for_activation_only(model, Variant.SPIDLU)
    records = apply_activation_surgery(model, Variant.SPIDLU, cfg())
    assert records
    names = trainable_parameter_names(model)
    assert names
    assert all(name.endswith("blend_alpha") for name in names)
    assert all(not param.requires_grad for name, param in model.named_parameters() if not name.endswith("blend_alpha"))


def test_spidlu_trainable_alpha_receives_gradients_and_updates():
    torch.manual_seed(123)
    model = FakeCausalLM(layers=2)
    freeze_pretrained_for_activation_only(model, Variant.SPIDLU)
    apply_activation_surgery(model, Variant.SPIDLU, cfg())
    names = trainable_parameter_names(model)
    assert names == [
        "model.layers.0.mlp.act_fn.blend_alpha",
        "model.layers.1.mlp.act_fn.blend_alpha",
    ]
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.0)
    group_summary = optimizer_parameter_group_summary(optimizer)
    assert group_summary == [{"group_index": 0, "parameter_count": 2, "element_count": 2, "lr": 1e-3, "weight_decay": 0.0}]
    before = trainable_parameter_snapshot(model)
    batch = next(iter(tiny_loader()))
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["input_ids"], labels=batch["labels"])
    alpha_regularizer = sum(param for _, param in model.named_parameters() if param.requires_grad)
    loss = output.loss - alpha_regularizer
    loss.backward()
    norms = gradient_norms(model)
    assert set(norms) == set(names)
    assert all(value is not None and value > 0 for value in norms.values())
    optimizer.step()
    clamp_trainable_blend_alphas(model)
    changed = changed_trainable_parameters(model, before)
    assert all(changed.values())
    assert all(0.0 <= param.item() <= cfg().spidlu_alpha_max for _, param in model.named_parameters() if param.requires_grad)


def test_compute_matched_freezes_pretrained_weights():
    model = FakeCausalLM(layers=2)
    freeze_pretrained_for_activation_only(model, Variant.ANN_COMPUTE_MATCHED)
    records = apply_activation_surgery(model, Variant.ANN_COMPUTE_MATCHED, cfg())
    assert records == []
    assert trainable_parameter_names(model) == []
    assert all(not param.requires_grad for param in model.parameters())


def test_compute_matched_training_preserves_weights_and_checkpoint(tmp_path):
    model = FakeCausalLM(layers=2)
    freeze_pretrained_for_activation_only(model, Variant.ANN_COMPUTE_MATCHED)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    stats = train_variant(model, tiny_loader(), cfg(), torch.device("cpu"), checkpoint_dir=tmp_path)
    after = model.state_dict()
    assert stats["optimizer_steps"] == 1
    assert stats["processed_tokens"] > 0
    assert stats["checkpoint_path"] is not None
    checkpoint = torch.load(stats["checkpoint_path"], map_location="cpu")
    assert checkpoint["optimizer_steps"] == 1
    assert checkpoint["processed_tokens"] == stats["processed_tokens"]
    assert "optimizer" not in checkpoint
    assert "scheduler" not in checkpoint
    assert all(torch.equal(before[name], after[name]) for name in before)
    reloaded = FakeCausalLM(layers=2)
    reloaded.load_state_dict(checkpoint["model"])
    assert all(torch.equal(after[name], reloaded.state_dict()[name]) for name in after)


def test_zero_step_diagnostic_forward_does_not_modify_base_weights():
    model = FakeCausalLM()
    before = state_fingerprint(model)
    model.eval()
    with torch.inference_mode():
        model(torch.randint(0, 17, (1, 6)), labels=torch.randint(0, 17, (1, 6)))
    after = state_fingerprint(model)
    assert before == after


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
        freeze_pretrained_for_activation_only(model, variant)
        apply_activation_surgery(model, variant, cfg())
        if variant != Variant.ANN_ORIGINAL:
            stats = train_variant(model, tiny_loader(), cfg(), torch.device("cpu"))
            assert stats["optimizer_steps"] == 1
        out = model(input_ids=torch.randint(0, 17, (1, 6)), labels=torch.randint(0, 17, (1, 6)))
        assert out.logits.shape[:2] == (1, 6)
