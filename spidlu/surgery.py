"""HuggingFace activation surgery for Phase 1 variants."""

from dataclasses import dataclass
from enum import Enum

import torch.nn as nn

from spidlu.layers import BlendedActivation, QuantizedActivationSTE, SpiDLU


class Variant(str, Enum):
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


def get_config_value(cfg, name, default=None):
    return getattr(cfg, name, default)


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


def _spidlu_activation(cfg):
    return SpiDLU(
        alpha=cfg.spidlu_alpha,
        threshold=cfg.spidlu_threshold,
        T=cfg.spidlu_T,
    )


def _replacement_for(variant, original, cfg):
    if variant == Variant.SPIDLU:
        spidlu = _spidlu_activation(cfg)
        if get_config_value(cfg, "spidlu_function_preserving", True):
            alpha_mode = get_config_value(cfg, "spidlu_alpha_mode", "trainable")
            alpha_max = get_config_value(cfg, "spidlu_alpha_max", 0.1)
            if alpha_max >= 1.0:
                raise ValueError("spidlu_alpha_max must be less than 1.0; direct alpha=1 jumps are disabled.")
            blend_alpha = 0.0
            trainable = alpha_mode == "trainable"
            if alpha_mode == "fixed":
                blend_alpha = get_config_value(cfg, "spidlu_fixed_alpha", 0.0)
                if blend_alpha > alpha_max:
                    raise ValueError("spidlu_fixed_alpha cannot exceed spidlu_alpha_max.")
            elif alpha_mode == "linear_warmup":
                blend_alpha = 0.0
            elif alpha_mode != "trainable":
                raise ValueError("spidlu_alpha_mode must be trainable, fixed, or linear_warmup.")
            return BlendedActivation(
                _clone_original_activation(original),
                spidlu,
                blend_alpha=blend_alpha,
                trainable=trainable,
                alpha_max=alpha_max,
                alpha_mode=alpha_mode,
            )
        return spidlu
    if variant == Variant.QUANTIZED_ACTIVATION:
        levels = cfg.quantized_levels or (cfg.spidlu_T + 1)
        return QuantizedActivationSTE(
            base_activation=_clone_original_activation(original),
            levels=levels,
        )
    raise ValueError(f"Variant {variant} does not replace activations.")


def selected_mlp_indices(total_layers, scope="all", layer_index=None, first_n=None):
    if scope == "all":
        return set(range(total_layers))
    if scope == "one":
        if layer_index is None:
            raise ValueError("surgery_layer_index is required when surgery_scope='one'.")
        if layer_index < 0 or layer_index >= total_layers:
            raise ValueError(f"surgery_layer_index {layer_index} is outside [0, {total_layers - 1}].")
        return {layer_index}
    if scope == "first_n":
        if first_n is None:
            raise ValueError("surgery_first_n is required when surgery_scope='first_n'.")
        if first_n < 1:
            raise ValueError("surgery_first_n must be at least 1.")
        return set(range(min(first_n, total_layers)))
    raise ValueError(f"Unknown surgery_scope: {scope}")


def apply_activation_surgery(model, variant, cfg):
    """Apply in-place activation replacement and return replacement records."""
    variant = Variant(variant)
    if variant in (Variant.ANN_ORIGINAL, Variant.ANN_COMPUTE_MATCHED):
        return []

    records = []
    mlps = list(iter_transformer_mlps(model))
    selected = selected_mlp_indices(
        len(mlps),
        scope=get_config_value(cfg, "surgery_scope", "all"),
        layer_index=get_config_value(cfg, "surgery_layer_index", None),
        first_n=get_config_value(cfg, "surgery_first_n", None),
    )
    for idx, path, mlp in mlps:
        if idx not in selected:
            continue
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


def freeze_pretrained_for_activation_only(model, variant):
    """Freeze pretrained weights for activation-only and compute-matched controls."""
    variant = Variant(variant)
    if variant not in (Variant.SPIDLU, Variant.QUANTIZED_ACTIVATION, Variant.ANN_COMPUTE_MATCHED):
        return
    for param in model.parameters():
        param.requires_grad = False


def trainable_parameter_names(model):
    return [name for name, param in model.named_parameters() if param.requires_grad]
