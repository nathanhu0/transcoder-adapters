"""Model registry for transcoder-adapted architectures.

Provides a unified interface to look up the correct Config and Model classes
for any supported architecture (qwen2, gemma2, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

# Architecture name -> (config class, model class) — populated lazily on first use
_REGISTRY: dict[str, tuple[type, type]] = {}


def _ensure_registered():
    """Lazily import and register all architectures on first access."""
    if _REGISTRY:
        return

    from models.qwen2_transcoder import (
        Qwen2ConfigWithTranscoder,
        Qwen2ForCausalLMWithTranscoder,
    )
    from models.gemma2_transcoder import (
        Gemma2ConfigWithTranscoder,
        Gemma2ForCausalLMWithTranscoder,
    )

    _REGISTRY["qwen2"] = (Qwen2ConfigWithTranscoder, Qwen2ForCausalLMWithTranscoder)
    _REGISTRY["gemma2"] = (Gemma2ConfigWithTranscoder, Gemma2ForCausalLMWithTranscoder)


def get_transcoder_classes(arch: str) -> tuple[type["PretrainedConfig"], type["PreTrainedModel"]]:
    """Return (ConfigWithTranscoder, ModelWithTranscoder) for the given architecture name."""
    _ensure_registered()
    if arch not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown architecture: '{arch}'. Available: {available}")
    return _REGISTRY[arch]


def detect_architecture(model_name: str) -> str:
    """Auto-detect architecture from a HuggingFace model name.

    Raises ValueError if the architecture cannot be determined.
    """
    name_lower = model_name.lower()
    if "qwen" in name_lower:
        return "qwen2"
    if "gemma" in name_lower:
        return "gemma2"
    raise ValueError(
        f"Cannot auto-detect architecture for '{model_name}'. "
        "Set 'model_arch' explicitly in your config (e.g. model_arch: qwen2)."
    )


def available_architectures() -> list[str]:
    """Return sorted list of registered architecture names."""
    _ensure_registered()
    return sorted(_REGISTRY.keys())
