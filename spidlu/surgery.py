"""HuggingFace activation surgery for Phase 1 variants."""

from dataclasses import dataclass
from enum import Enum

import torch.nn as nn

from spidlu.layers import QuantizedActivationSTE, SpiDLU


class Variant(Enum):
    ANN_ORIGINAL = "ann_original"
    SPIDLU = "spidlu"
    ANN_COMPUTE_MATCHED = "ann_compute_matched"
    QUANTIZED_ACTIVATION = "quantized_activation"


@dataclass
class ReplacementRecord:
    layer_index: int
    module_path: str
    activation_attr: str
    original_type: str
    replacement_type: str
    semantic_location: str


def _base_model(model):
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    return getattr(model, "model", getattr(model, "transformer", model))


def iter_transformer_mlps(model):
    """Yield (index, path, mlp) for common HF causal-LM architectures."""
    base = _base_model(model)
    candidates = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("base.layers", getattr(base, "layers", None)),
        ("transformer.h", getattr(getattr(model, "transformer", None), "h", None)),
        ("base.h", getattr(base, "h", None)),
        ("gpt_neox.layers", getattr(getattr(model, "gpt_neox", None), "layers", None)),
    ]
    for prefix, layers in candidates:
        if layers is None:
            continue
        for idx, block in enumerate(layers):
            mlp = getattr(block, "mlp", None)
            if mlp is not None:
                yield idx, f"{prefix}.{idx}.mlp", mlp
        return
    raise ValueError("Could not locate transformer MLP layers for activation surgery.")


def _activation_attr(mlp):
    # Llama/Qwen/Mistral use act_fn in down_proj(act_fn(gate_proj(x)) * up_proj(x)).
    # GPT-style MLPs commonly expose act; Phi variants may expose activation_fn.
    for attr in ("act_fn", "activation_fn", "activation", "act"):
        if hasattr(mlp, attr):
            return attr
    raise ValueError(f"Could not locate activation attribute on {type(mlp).__name__}.")


def _semantic_location(mlp, activation_attr):
    if all(hasattr(mlp, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
        return f"down_proj({activation_attr}(gate_proj(x)) * up_proj(x))"
    if all(hasattr(mlp, attr) for attr in ("fc1", "fc2")):
        return f"fc2({activation_attr}(fc1(x)))"
    if all(hasattr(mlp, attr) for attr in ("c_fc", "c_proj")):
        return f"c_proj({activation_attr}(c_fc(x)))"
    return f"{type(mlp).__name__}.{activation_attr}"


def _clone_original_activation(original):
    if isinstance(original, nn.Module):
        return original
    # HF activations are often callable objects. Keep them callable inside a Module.
    class CallableActivation(nn.Module):
        def forward(self, x):
            return original(x)

    return CallableActivation()


def _replacement_for(variant, original, cfg):
    if variant == Variant.SPIDLU:
        return SpiDLU(
            alpha=cfg.spidlu_alpha,
            threshold=cfg.spidlu_threshold,
            T=cfg.spidlu_T,
        )
    if variant == Variant.QUANTIZED_ACTIVATION:
        levels = cfg.quantized_levels or (cfg.spidlu_T + 1)
        return QuantizedActivationSTE(
            base_activation=_clone_original_activation(original),
            levels=levels,
        )
    raise ValueError(f"Variant {variant.name} does not replace activations.")


def apply_activation_surgery(model, variant, cfg):
    """Apply in-place activation replacement and return replacement records."""
    variant = Variant(variant)
    if variant in (Variant.ANN_ORIGINAL, Variant.ANN_COMPUTE_MATCHED):
        return []

    records = []
    for idx, path, mlp in iter_transformer_mlps(model):
        attr = _activation_attr(mlp)
        original = getattr(mlp, attr)
        replacement = _replacement_for(variant, original, cfg)
        setattr(mlp, attr, replacement)
        records.append(
            ReplacementRecord(
                layer_index=idx,
                module_path=path,
                activation_attr=attr,
                original_type=type(original).__name__,
                replacement_type=type(replacement).__name__,
                semantic_location=_semantic_location(mlp, attr),
            )
        )
    if not records:
        raise ValueError("No activation replacements were applied.")
    return records
