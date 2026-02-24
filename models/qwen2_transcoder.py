"""
Qwen2 model with integrated transcoder adapters.

This module provides a Qwen2 model class that has transcoder layers built into
the MLP architecture, allowing it to be saved and loaded as a standard
HuggingFace checkpoint without requiring separate adapter loading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections.abc import Iterator
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


class Qwen2ConfigWithTranscoder(Qwen2Config):
    """Qwen2 config with transcoder parameters."""

    model_type = "qwen2"

    def __init__(self, transcoder_n_features=512, transcoder_dec_bias=False, **kwargs):
        """Initialize config with transcoder parameters.

        Args:
            transcoder_n_features: Number of features in transcoder
            transcoder_dec_bias: Whether to include bias in decoder
            **kwargs: Other Qwen2Config parameters
        """
        self.transcoder_n_features = transcoder_n_features
        self.transcoder_dec_bias = transcoder_dec_bias
        super().__init__(**kwargs)

        # Override architectures for vLLM compatibility
        self.architectures = ["Qwen2ForCausalLMWithTranscoder"]


class Qwen2MLPWithTranscoder(Qwen2MLP):
    """Qwen2 MLP with integrated transcoder branch."""

    def __init__(self, config):
        """Initialize MLP with integrated transcoder."""
        # Initialize the original MLP layers
        super().__init__(config)

        self.config = config
        self.d_model = config.hidden_size
        self.n_features = getattr(config, 'transcoder_n_features', 512)
        self.dec_bias = getattr(config, 'transcoder_dec_bias', False)

        # Create transcoder layers: d_model -> n_features -> d_model
        self.transcoder_enc = nn.Linear(self.d_model, self.n_features, bias=True)
        self.transcoder_dec = nn.Linear(self.n_features, self.d_model, bias=self.dec_bias)

        # Initialize transcoder weights to ensure zero initial contribution
        self._init_transcoder_weights()

        # Flag to disable transcoder for clean forward passes
        self.disable_transcoder = False

        # Training state: set cache_features=True to compute and cache L1/stats
        # during forward. Only enable for the main forward pass, not bridging/NMSE.
        self.cache_features = False
        self.cached_l1 = None       # scalar tensor with grad (for loss)
        self.cached_l0 = None       # float (for logging)
        # Dead feature tracking: persistent counter per feature, never cleared.
        # Increments by batch_size each forward, resets to 0 for active features.
        self._dead_feature_counters = torch.zeros(self.n_features)

    def _init_transcoder_weights(self):
        """Initialize transcoder weights."""
        scale_d_model = 1 / math.sqrt(self.d_model)
        nn.init.uniform_(self.transcoder_enc.weight, -scale_d_model, scale_d_model)
        nn.init.zeros_(self.transcoder_enc.bias)
        # Initialize decoder to zero (ensures initial contribution is zero)
        nn.init.zeros_(self.transcoder_dec.weight)
        if self.dec_bias:
            nn.init.zeros_(self.transcoder_dec.bias)

    def forward(self, hidden_states):  # type: ignore[override]
        """Forward pass with original MLP + transcoder branch."""
        # Original MLP computation (from parent class)
        original_output = super().forward(hidden_states)

        # Allow disabling transcoder for clean forward passes
        if getattr(self, 'disable_transcoder', False):
            return original_output

        # Transcoder computation: f = ReLU(W_enc * x + b_enc), y = W_dec * f
        features = F.relu(self.transcoder_enc(hidden_states))  # [batch, seq, n_features]
        transcoder_output = self.transcoder_dec(features)      # [batch, seq, d_model]

        if self.cache_features:
            batch_size = features.shape[0]

            # L1: weighted by decoder column norms, sum over features, mean over (batch, seq)
            # Differentiable — stays in computation graph for backward
            dec_column_norms = torch.norm(self.transcoder_dec.weight, dim=0)  # [n_features]
            weighted_features = features * dec_column_norms.unsqueeze(0).unsqueeze(0)
            self.cached_l1 = weighted_features.sum(dim=-1).mean()

            with torch.no_grad():
                feature_active = features > 0  # [batch, seq, n_features]

                # L0: count active features per token, mean over (batch, seq)
                self.cached_l0 = feature_active.float().sum(dim=-1).mean().item()

                # Dead features: age all, reset active ones
                self._dead_feature_counters = self._dead_feature_counters.to(features.device)
                self._dead_feature_counters += batch_size
                self._dead_feature_counters[feature_active.any(dim=(0, 1))] = 0

        return original_output + transcoder_output


class Qwen2ForCausalLMWithTranscoder(Qwen2ForCausalLM):
    """Qwen2 causal LM with integrated transcoder adapters."""

    config_class = Qwen2ConfigWithTranscoder

    def __init__(self, config):
        """Initialize model with transcoder-equipped MLPs."""
        super().__init__(config)

        # Replace all MLP modules with transcoder versions
        for layer in self.model.layers:
            layer.mlp = Qwen2MLPWithTranscoder(config)

    def _transcoder_mlps(self) -> Iterator[Qwen2MLPWithTranscoder]:
        """Yield each transcoder MLP layer with proper typing."""
        for layer in self.model.layers:
            yield layer.mlp  # type: ignore[misc]

    def set_cache_features(self, enabled: bool):
        """Toggle feature caching on all transcoder layers."""
        for mlp in self._transcoder_mlps():
            mlp.cache_features = enabled

    def collect_sparsity_loss(self) -> torch.Tensor:
        """Sum raw L1 loss across all layers. Differentiable. Caller applies l1_weight."""
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        for mlp in self._transcoder_mlps():
            if mlp.cached_l1 is not None:
                total = total + mlp.cached_l1.to(device)
        return total

    def collect_transcoder_stats(self) -> dict:
        """Collect L0 and dead feature stats across all layers for logging."""
        stats = {}
        l0_values = []
        dead_100_total = 0
        dead_1000_total = 0
        for i, mlp in enumerate(self._transcoder_mlps()):
            if mlp.cached_l0 is not None:
                l0_values.append(mlp.cached_l0)
                stats[f"adapter_stats_per_layer/layer_{i}_l0_count"] = mlp.cached_l0
            dead_100 = (mlp._dead_feature_counters >= 100).sum().item()
            dead_1000 = (mlp._dead_feature_counters >= 1000).sum().item()
            dead_100_total += dead_100
            dead_1000_total += dead_1000
            stats[f"adapter_stats_per_layer/layer_{i}_dead_features_100"] = dead_100
            stats[f"adapter_stats_per_layer/layer_{i}_dead_features_1000"] = dead_1000
        n_layers = len(self.model.layers)
        if l0_values:
            stats["adapter_stats_avg/l0_count"] = sum(l0_values) / len(l0_values)
        stats["adapter_stats_avg/dead_features_100"] = dead_100_total / n_layers
        stats["adapter_stats_avg/dead_features_1000"] = dead_1000_total / n_layers
        return stats

    def clear_cached_stats(self):
        """Clear per-step caches (L1/L0). Dead feature counters are persistent."""
        for mlp in self._transcoder_mlps():
            mlp.cached_l1 = None
            mlp.cached_l0 = None


def register_qwen2_transcoder():
    """Register Qwen2 transcoder model with transformers AutoModel."""
    import transformers
    setattr(transformers, "Qwen2ForCausalLMWithTranscoder", Qwen2ForCausalLMWithTranscoder)


register_qwen2_transcoder()