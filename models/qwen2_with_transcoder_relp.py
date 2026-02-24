"""
Qwen2 model with RelP-aware backward pass for attribution analysis.

Load the same weights as Qwen2ForCausalLMWithTranscoder but with modified
backward passes that implement Relevance Propagation rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterator

from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast



class Qwen2RMSNormRelP(nn.Module):
    """RMSNorm with RelP-aware backward pass."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.relp_enabled = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Only upcast to float32 for numerical stability (don't downcast from float64)
        if input_dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(torch.float32)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        if self.relp_enabled:
            # RelP: freeze normalization factor
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon).detach()
        else:
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


class Qwen2MLPWithTranscoderRelP(nn.Module):
    """MLP + Transcoder with RelP-aware backward pass."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Base MLP layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Transcoder layers
        self.n_features = getattr(config, 'transcoder_n_features', 512)
        self.dec_bias = getattr(config, 'transcoder_dec_bias', False)
        self.transcoder_enc = nn.Linear(self.hidden_size, self.n_features, bias=True)
        self.transcoder_dec = nn.Linear(self.n_features, self.hidden_size, bias=self.dec_bias)

        # RelP settings (set on model, propagated here)
        self.relp_enabled = True
        self.stop_grad_at: set[str] = set()

        # Optional: disable transcoder entirely
        self.disable_transcoder = False

        # Cached activations for attribution (populated during forward)
        self.cached_features: torch.Tensor | None = None

        # Feature mask for ablation: [n_features], 1=keep, 0=suppress
        self.feature_mask: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # === Base MLP ===
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)

        if self.relp_enabled:
            # RelP SiLU: x * detach(sigmoid(x))
            gate_act = gate * torch.sigmoid(gate).detach()
            # RelP half rule: 0.5 * combined + 0.5 * detach(combined)
            combined = gate_act * up
            combined = 0.5 * combined + 0.5 * combined.detach()
        else:
            gate_act = F.silu(gate)
            combined = gate_act * up

        base_output = self.down_proj(combined)

        # Optional stop_grad at base MLP output
        if "base_mlp_output" in self.stop_grad_at:
            base_output = base_output.detach()

        # === Transcoder ===
        if self.disable_transcoder:
            self.cached_features = None
            return base_output

        pre_act = self.transcoder_enc(hidden_states)
        features = F.relu(pre_act)

        # Apply feature mask for ablation (if set)
        if self.feature_mask is not None:
            features = features * self.feature_mask

        # Handle stop_grad at transcoder features
        if "transcoder_features" in self.stop_grad_at:
            # Make features a leaf node that tracks gradients
            features = features.detach().requires_grad_(True)
        elif torch.is_grad_enabled() and features.requires_grad:
            # Keep in graph but retain grad for attribution (only if grad enabled)
            features.retain_grad()

        # Cache for later attribution access
        self.cached_features = features

        transcoder_output = self.transcoder_dec(features)

        # Optional stop_grad at transcoder output
        if "transcoder_output" in self.stop_grad_at:
            transcoder_output = transcoder_output.detach()

        return base_output + transcoder_output


class Qwen2AttentionRelP(nn.Module):
    """Attention with RelP-aware backward pass."""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # RelP settings
        self.relp_enabled = True
        self.stop_grad_at: set[str] = set()

        # Memory-efficient chunked attention (default on)
        self.use_chunked_attention = True
        self.attention_chunk_size = 512

    def _chunked_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """Memory-efficient chunked attention for RelP.

        Processes queries in chunks to reduce peak memory from O(seq^2) to O(seq * chunk).
        Since RelP detaches attention weights, we don't need to store them for backward.

        Args:
            query_states: [B, H, S, D]
            key_states: [B, H, S, D]
            value_states: [B, H, S, D]

        Returns:
            attn_output: [B, H, S, D]
        """
        B, H, S, D = query_states.shape
        chunk_size = self.attention_chunk_size

        outputs = []
        for chunk_start in range(0, S, chunk_size):
            chunk_end = min(chunk_start + chunk_size, S)
            q_chunk = query_states[:, :, chunk_start:chunk_end]  # [B, H, chunk, D]

            # Causal: attend to positions 0..chunk_end-1
            k_slice = key_states[:, :, :chunk_end]
            v_slice = value_states[:, :, :chunk_end]

            # Compute attention scores for this chunk: [B, H, chunk, chunk_end]
            scores = torch.matmul(q_chunk, k_slice.transpose(-2, -1)) * self.scaling

            # Causal mask: position i in chunk attends to positions 0..(chunk_start + i)
            chunk_len = chunk_end - chunk_start
            kv_len = chunk_end
            # Create mask where valid positions are True
            row_idx = torch.arange(chunk_len, device=query_states.device).unsqueeze(1)
            col_idx = torch.arange(kv_len, device=query_states.device).unsqueeze(0)
            causal_mask = col_idx <= (row_idx + chunk_start)
            scores = scores.masked_fill(~causal_mask, float('-inf'))

            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # RelP: detach attention weights
            if self.relp_enabled:
                chunk_out = torch.matmul(weights.detach(), v_slice)
            else:
                chunk_out = torch.matmul(weights, v_slice)

            outputs.append(chunk_out)

        return torch.cat(outputs, dim=2)

    def _full_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple:
        """Original full attention (for debugging/comparison)."""
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # RelP: freeze attention weights
        if self.relp_enabled:
            attn_output = torch.matmul(attn_weights.detach(), value_states)
        else:
            attn_output = torch.matmul(attn_weights, value_states)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple:
        bsz, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Choose attention implementation
        if self.use_chunked_attention:
            attn_output = self._chunked_attention(query_states, key_states, value_states)
            attn_weights = None  # Not computed/stored in chunked mode
        else:
            attn_output, attn_weights = self._full_attention(
                query_states, key_states, value_states, attention_mask
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        # Optional stop_grad at attention output
        if "attn_output" in self.stop_grad_at:
            attn_output = attn_output.detach()

        return attn_output, attn_weights


class Qwen2DecoderLayerRelP(nn.Module):
    """Decoder layer with RelP-aware components."""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2AttentionRelP(config, layer_idx)
        self.mlp = Qwen2MLPWithTranscoderRelP(config)
        self.input_layernorm = Qwen2RMSNormRelP(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNormRelP(config.hidden_size, eps=config.rms_norm_eps)

        # For compatibility with mask selection
        layer_types = getattr(config, 'layer_types', None)
        self.attention_type = layer_types[layer_idx] if layer_types is not None else "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2ModelRelP(nn.Module):
    """Qwen2 model backbone with RelP-aware layers."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayerRelP(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen2RMSNormRelP(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            assert inputs_embeds is not None

        hidden_states = inputs_embeds
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Build causal mask (match dtype of hidden_states)
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
        else:
            causal_mask = attention_mask

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Qwen2ForCausalLMWithTranscoderRelP(nn.Module):
    """
    Qwen2 + Transcoder with RelP-aware backward pass.

    Usage:
        model = Qwen2ForCausalLMWithTranscoderRelP.from_pretrained(path)
        model.set_stop_grad_at({"transcoder_features"})  # direct effect only

        output = model(input_ids)
        output.logits[0, -1, target_token].backward()
        # Gradients now follow RelP rules
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2ModelRelP(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # RelP settings
        self.relp_enabled = True
        self.stop_grad_at: set[str] = set()

    def set_relp_enabled(self, enabled: bool):
        """Enable/disable RelP rules globally."""
        self.relp_enabled = enabled
        self._propagate_relp_settings()

    def set_stop_grad_at(self, points: set[str]):
        """Set which points should have gradients stopped."""
        self.stop_grad_at = points
        self._propagate_relp_settings()

    def _decoder_layers(self) -> Iterator[Qwen2DecoderLayerRelP]:
        """Yield each decoder layer with proper typing."""
        for layer in self.model.layers:
            yield layer  # type: ignore[misc]

    def _propagate_relp_settings(self):
        """Propagate RelP settings to all submodules."""
        for layer in self._decoder_layers():
            layer.input_layernorm.relp_enabled = self.relp_enabled
            layer.post_attention_layernorm.relp_enabled = self.relp_enabled
            layer.self_attn.relp_enabled = self.relp_enabled
            layer.self_attn.stop_grad_at = self.stop_grad_at
            layer.mlp.relp_enabled = self.relp_enabled
            layer.mlp.stop_grad_at = self.stop_grad_at
        self.model.norm.relp_enabled = self.relp_enabled  # type: ignore[union-attr]

    def set_chunked_attention(self, enabled: bool, chunk_size: int = 512):
        """Enable/disable memory-efficient chunked attention.

        Args:
            enabled: If True, use chunked attention (O(seq*chunk) memory).
                     If False, use full attention (O(seq^2) memory).
            chunk_size: Size of query chunks (default 512).
        """
        for layer in self._decoder_layers():
            layer.self_attn.use_chunked_attention = enabled
            layer.self_attn.attention_chunk_size = chunk_size

    def set_feature_mask(self, mask: torch.Tensor | None):
        """Set feature mask for ablation studies.

        Args:
            mask: Shape [n_layers, n_features], 1=keep, 0=suppress.
                  Pass None to clear all masks.
        """
        for i, layer in enumerate(self._decoder_layers()):
            layer.mlp.feature_mask = mask[i] if mask is not None else None

    def get_cached_features(self) -> dict:
        """Get cached feature activations for all layers.

        Returns:
            Dict mapping layer_idx -> features tensor [batch, seq, n_features]
        """
        return {
            i: layer.mlp.cached_features
            for i, layer in enumerate(self._decoder_layers())
            if layer.mlp.cached_features is not None
        }

    def get_feature_attributions(self) -> dict:
        """Get feature attributions (features * grad) for all layers.

        Call this after backward() to get per-feature attributions.

        Returns:
            Dict mapping layer_idx -> attribution tensor [batch, seq, n_features]
        """
        attrs = {}
        for i, layer in enumerate(self._decoder_layers()):
            f = layer.mlp.cached_features
            if f is not None and f.grad is not None:
                attrs[i] = f * f.grad
        return attrs

    def get_feature_attribution_summary(self) -> dict:
        """Get summed feature attributions per layer.

        Returns:
            Dict mapping layer_idx -> scalar total attribution for that layer
        """
        attrs = self.get_feature_attributions()
        return {i: attr.sum().item() for i, attr in attrs.items()}

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        return CausalLMOutputWithPast(logits=logits)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load weights from a local checkpoint or HuggingFace repo ID.

        Args:
            path: Local directory, local file, or HF repo ID
                  (e.g. "nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4")
        """
        from transformers import AutoConfig
        import os

        # Resolve HF repo ID to local path
        if not os.path.exists(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path)

        # Load config
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        assert isinstance(config, Qwen2Config)

        # Create model
        model = cls(config)

        # Load state dict
        if os.path.isdir(path):
            import glob
            weight_files = glob.glob(os.path.join(path, "*.safetensors"))
            if weight_files:
                from safetensors.torch import load_file
                state_dict = {}
                for f in weight_files:
                    state_dict.update(load_file(f))
            else:
                weight_file = os.path.join(path, "pytorch_model.bin")
                state_dict = torch.load(weight_file, map_location="cpu")
        else:
            state_dict = torch.load(path, map_location="cpu")

        # Load weights (may need key remapping)
        model.load_state_dict(state_dict, strict=False)

        return model
