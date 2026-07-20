from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spidlu.layers import QuantizedActivationSTE, SpiDLU
from spidlu.metrics import count_parameters
from spidlu.phase1 import state_fingerprint
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


def test_spidlu_replaces_gated_activation_location():
    model = FakeCausalLM()
    records = apply_activation_surgery(model, Variant.SPIDLU, cfg())
    assert records
    assert isinstance(model.model.layers[0].mlp.act_fn, SpiDLU)
    assert records[0].semantic_location == "down_proj(act_fn(gate_proj(x)) * up_proj(x))"


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
