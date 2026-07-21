"""Activation modules used by Phase 1 variants."""

import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    functional = None
    neuron = None
    surrogate = None


class SpiDLU(nn.Module):
    """Spiking discrete linear unit represented as a mean firing rate."""

    def __init__(self, alpha=0.9, threshold=1.0, T=4):
        super().__init__()
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0, 1).")
        if T < 1:
            raise ValueError("T must be at least 1.")
        self.alpha = alpha
        self.threshold = threshold
        self.T = int(T)
        if neuron is not None:
            self.lif = neuron.LIFNode(
                tau=1.0 / (1 - alpha),
                v_threshold=threshold,
                surrogate_function=surrogate.Sigmoid(),
            )
        else:
            self.lif = None

    def forward(self, x):
        if self.lif is None:
            # Lightweight fallback keeps tests importable without SpikingJelly.
            return torch.sigmoid((x - self.threshold) / max(1e-6, 1 - self.alpha))
        functional.reset_net(self.lif)
        repeat_shape = (self.T,) + (1,) * x.dim()
        x_seq = x.unsqueeze(0).repeat(repeat_shape)
        spikes = [self.lif(x_seq[t]) for t in range(self.T)]
        out = torch.stack(spikes).mean(0)
        functional.reset_net(self.lif)
        return out


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class QuantizedActivationSTE(nn.Module):
    """Non-spiking activation control with straight-through quantization."""

    def __init__(self, base_activation=None, levels=5, clip_min=-8.0, clip_max=8.0):
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be at least 2.")
        self.base_activation = base_activation if base_activation is not None else nn.SiLU()
        self.levels = int(levels)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def forward(self, x):
        y = self.base_activation(x)
        clipped = torch.clamp(y, self.clip_min, self.clip_max)
        scaled = (clipped - self.clip_min) / (self.clip_max - self.clip_min)
        quantized = _RoundSTE.apply(scaled * (self.levels - 1)) / (self.levels - 1)
        return quantized * (self.clip_max - self.clip_min) + self.clip_min


class BlendedActivation(nn.Module):
    """Blend a reference activation with a replacement activation.

    With blend_alpha=0 this is exactly the reference activation. This is useful
    for function-preserving surgery diagnostics and staged alignment.
    """

    def __init__(self, reference_activation, replacement_activation, blend_alpha=0.0, trainable=False):
        super().__init__()
        if not 0.0 <= blend_alpha <= 1.0:
            raise ValueError("blend_alpha must be in [0, 1].")
        self.reference_activation = reference_activation
        self.replacement_activation = replacement_activation
        alpha = torch.tensor(float(blend_alpha))
        if trainable:
            self.blend_alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("blend_alpha", alpha)

    def forward(self, x):
        alpha = self.blend_alpha.clamp(0.0, 1.0)
        return (1.0 - alpha) * self.reference_activation(x) + alpha * self.replacement_activation(x)
