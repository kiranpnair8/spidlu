"""Phase 1 RQ1 utilities for Spi-dLU language-model experiments."""

from spidlu.layers import BlendedActivation, QuantizedActivationSTE, SpiDLU
from spidlu.surgery import Variant, apply_activation_surgery

__all__ = ["BlendedActivation", "QuantizedActivationSTE", "SpiDLU", "Variant", "apply_activation_surgery"]
