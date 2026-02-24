"""
vLLM-compatible Qwen2 model with integrated transcoder adapters.

This module provides a vLLM model that inherits from the native vLLM Qwen2ForCausalLM
and adds transcoder layers to the MLP modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections.abc import Iterable
from typing import Any

from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM as VLLMQwen2ForCausalLM, Qwen2MLP
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.config import VllmConfig


class Qwen2MLPWithTranscoder(Qwen2MLP):
    """vLLM Qwen2 MLP with integrated transcoder adapter."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Any | None = None,
        prefix: str = "",
        transcoder_n_features: int = 512,
        transcoder_dec_bias: bool = False,
    ):
        """Initialize MLP with integrated transcoder."""
        # Initialize the original vLLM MLP
        super().__init__(hidden_size, intermediate_size, hidden_act, quant_config, prefix)

        self.hidden_size = hidden_size
        self.n_features = transcoder_n_features
        self.dec_bias = transcoder_dec_bias

        # Create transcoder layers: d_model -> n_features -> d_model
        # Note: Using simple Linear layers for now, could use parallel versions later
        self.transcoder_enc = nn.Linear(self.hidden_size, self.n_features, bias=True)
        self.transcoder_dec = nn.Linear(self.n_features, self.hidden_size, bias=transcoder_dec_bias)

        # Store prefix for weight loading
        self.prefix = prefix

        # Initialize transcoder weights to ensure zero initial contribution
        self._init_transcoder_weights()

    def _init_transcoder_weights(self):
        """Initialize transcoder weights."""
        scale_d_model = 1 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.transcoder_enc.weight, -scale_d_model, scale_d_model)
        nn.init.zeros_(self.transcoder_enc.bias)
        # Initialize decoder to zero (ensures initial contribution is zero)
        nn.init.zeros_(self.transcoder_dec.weight)
        if self.dec_bias:
            nn.init.zeros_(self.transcoder_dec.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with original MLP + transcoder branch."""
        # Original MLP computation (from parent class)
        original_output = super().forward(x)

        # Transcoder computation: f = ReLU(W_enc * x + b_enc), y = W_dec * f
        pre_activations = self.transcoder_enc(x)       # [batch, seq, n_features]
        features = F.relu(pre_activations)             # [batch, seq, n_features]
        transcoder_output = self.transcoder_dec(features)  # [batch, seq, d_model]

        # Combine outputs
        return original_output + transcoder_output


class Qwen2ForCausalLMWithTranscoder(VLLMQwen2ForCausalLM):
    """vLLM Qwen2 causal LM with integrated transcoder adapters."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """Initialize model with transcoder-equipped MLPs."""
        # Initialize the parent vLLM model
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Replace all MLP modules with transcoder versions
        config = vllm_config.model_config.hf_config
        quant_config = getattr(vllm_config, 'quant_config', None)
        transcoder_n_features = getattr(config, 'transcoder_n_features', 512)

        for i, layer in enumerate(self.model.layers):
            # Replace the MLP with our transcoder version
            layer.mlp = Qwen2MLPWithTranscoder(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.model.layers.{i}.mlp",
                transcoder_n_features=transcoder_n_features,
                transcoder_dec_bias=getattr(config, 'transcoder_dec_bias', False),
            )

    def load_weights(self, weights: "Iterable[tuple[str, torch.Tensor]]") -> "set[str]":
        """Load weights including transcoder adapter weights."""
        # Separate transcoder weights from model weights
        model_weights = []
        transcoder_weights = {}

        for name, weight in weights:
            if 'transcoder_enc' in name or 'transcoder_dec' in name:
                transcoder_weights[name] = weight
            else:
                model_weights.append((name, weight))

        # Load model weights first using parent method
        loaded_weights = set()
        if model_weights:
            loaded_weights = super().load_weights(model_weights)

        # TODO: test vLLM weight loading after refactor
        # Load transcoder weights by direct parameter name mapping
        param_dict = dict(self.named_parameters())
        for name, weight in transcoder_weights.items():
            if name in param_dict:
                default_weight_loader(param_dict[name], weight)
                loaded_weights.add(name)

        return loaded_weights


def register_vllm_transcoder():
    """Register the transcoder model with vLLM."""
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "Qwen2ForCausalLMWithTranscoder",
        "models.qwen2_transcoder_vllm:Qwen2ForCausalLMWithTranscoder"
    )