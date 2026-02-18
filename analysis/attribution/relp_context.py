"""
Attribution context for RelP models using feature.grad collection.

Key difference from standard context:
- Edge weights computed via backward through RelP-linearized model
- Collect feature.grad instead of dot products with decoder vecs
- No error nodes - MLP influence folded into feature edges
"""

from typing import TYPE_CHECKING
from functools import partial

import torch
from einops import einsum

if TYPE_CHECKING:
    from analysis.attribution.relp_model import RelPReplacementModel


class RelPAttributionContext:
    """
    Manages attribution computation for RelP models.

    Node layout (columns): [features | tokens]
    - No error nodes - base MLP influence is captured in feature edges via RelP

    The key insight: with stop_grad at features, backward through the RelP
    model gives us gradients that capture both direct transcoder paths AND
    paths mediated through the base MLP.
    """

    def __init__(
        self,
        model: "RelPReplacementModel",
        tokens: torch.Tensor,
        activation_matrix: torch.Tensor,
        encoder_vecs: torch.Tensor,
        token_vectors: torch.Tensor,
        logits: torch.Tensor,
    ):
        self.model = model
        self.tokens = tokens
        self.activation_matrix = activation_matrix
        self.encoder_vecs = encoder_vecs  # (n_active_features, d_model)
        self.token_vectors = token_vectors  # (n_pos, d_model)
        self.logits = logits

        # Dimensions
        self.n_layers = model.cfg.n_layers
        self.n_pos = len(tokens)
        self.n_features = activation_matrix._nnz()
        self.n_tokens = self.n_pos

        # Row size for edge matrix: [features | tokens]
        self._row_size = self.n_features + self.n_tokens

        # Cache for residual activations (filled during forward)
        self._resid_cache: dict[int, torch.Tensor] = {}

        # Batch buffer for accumulating edge weights
        self._batch_buffer: torch.Tensor | None = None

    def _run_forward_with_cache(self, batch_size: int) -> torch.Tensor:
        """
        Run forward pass, caching residual activations for gradient injection.

        Returns the final hidden states (before lm_head).
        """
        model = self.model.model  # The underlying Qwen2ForCausalLMWithTranscoderRelP

        # IMPORTANT: Set stop_grad BEFORE forward so features become leaf nodes
        model.set_stop_grad_at({"transcoder_features"})

        # Expand tokens to batch
        input_ids = self.tokens.unsqueeze(0).expand(batch_size, -1)

        # Get embeddings
        hidden_states = model.model.embed_tokens(input_ids)
        hidden_states.retain_grad()  # Need this to capture embedding gradients
        self._resid_cache['embed'] = hidden_states

        # Setup position embeddings
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        # Build causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'),
                      device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # Forward through layers, caching residuals
        for layer_idx, layer in enumerate(model.model.layers):
            # Cache residual before this layer (for gradient injection)
            hidden_states.retain_grad()  # Need this to capture gradients at each layer
            self._resid_cache[layer_idx] = hidden_states

            # Run layer
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )

        # Final norm
        hidden_states = model.model.norm(hidden_states)
        hidden_states.retain_grad()
        self._resid_cache[self.n_layers] = hidden_states

        return hidden_states

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """
        Compute attribution rows for a batch of target nodes.

        For each target at (layer, pos), we:
        1. Inject gradient (inject_values) at that location
        2. Backward through RelP model with stop_grad at features
        3. Collect feature.grad for all upstream features
        4. Collect embedding contributions

        Args:
            layers: (batch,) layer indices for targets
            positions: (batch,) position indices for targets
            inject_values: (batch, d_model) gradient vectors to inject
            retain_graph: whether to retain graph for more backward passes

        Returns:
            (batch, row_size) matrix of edge weights
        """
        batch_size = len(layers)
        device = self.model.cfg.device
        dtype = self.model.cfg.dtype

        # Run forward if cache is empty
        if not self._resid_cache:
            self._run_forward_with_cache(batch_size)

        # Zero existing gradients from previous batch
        self._zero_grads()

        # Prepare output buffer
        self._batch_buffer = torch.zeros(
            self._row_size, batch_size,
            dtype=dtype, device=device
        )

        # Create gradient tensor to inject
        # We inject at the residual stream at the target layer
        unique_layers = layers.unique().tolist()

        # Register hooks to inject gradients
        handles = []
        batch_idx = torch.arange(batch_size, device=device)

        for layer_idx in unique_layers:
            mask = layers == layer_idx
            if not mask.any():
                continue

            target_positions = positions[mask]
            target_values = inject_values[mask]
            target_batch_idx = batch_idx[mask]

            # Get cached residual for this layer
            resid = self._resid_cache[layer_idx]

            # Create hook to inject gradient
            def make_inject_hook(b_idx, p_idx, vals):
                def hook(grad):
                    grad_out = grad.clone()
                    # Inject at specific (batch, position) locations
                    for i, (b, p, v) in enumerate(zip(b_idx, p_idx, vals)):
                        grad_out[b, p] += v.to(grad_out.dtype)
                    return grad_out
                return hook

            h = resid.register_hook(
                make_inject_hook(target_batch_idx, target_positions, target_values)
            )
            handles.append(h)

        # Backward from the highest layer in batch
        try:
            max_layer = max(unique_layers)
            resid = self._resid_cache[max_layer]

            # Backward with zero gradient (hooks inject the actual gradient)
            resid.backward(
                gradient=torch.zeros_like(resid),
                retain_graph=retain_graph
            )
        finally:
            for h in handles:
                h.remove()

        # Collect feature gradients
        self._collect_feature_grads(batch_size)

        # Collect token/embedding contributions
        self._collect_token_grads(batch_size)

        # Return transposed buffer
        result = self._batch_buffer.T[:batch_size].clone()
        self._batch_buffer = None

        return result

    def _collect_feature_grads(self, batch_size: int):
        """Collect gradients from cached features into batch buffer."""
        if self.n_features == 0:
            return

        feat_layers, feat_pos, feat_idx = self.activation_matrix.indices()
        activation_values = self.activation_matrix.values()

        for i, (layer, pos, feat) in enumerate(zip(feat_layers, feat_pos, feat_idx)):
            layer = layer.item()
            pos = pos.item()
            feat = feat.item()

            cached = self.model.model.model.layers[layer].mlp.cached_features
            if cached is None or cached.grad is None:
                continue

            # cached.grad shape: (batch, seq, n_features)
            # Get gradient for this specific feature at this position
            grad_vals = cached.grad[:batch_size, pos, feat]  # (batch,)

            # Edge weight = activation * grad (input x grad attribution)
            act_val = activation_values[i]
            self._batch_buffer[i] = grad_vals * act_val

    def _collect_token_grads(self, batch_size: int):
        """Collect embedding gradients for token contributions."""
        embed_resid = self._resid_cache.get('embed')
        if embed_resid is None or embed_resid.grad is None:
            return

        # embed_resid.grad: (batch, seq, d_model)
        # token_vectors: (seq, d_model)
        # Contribution = dot(grad, embedding) for each position

        for pos in range(self.n_pos):
            grad_at_pos = embed_resid.grad[:batch_size, pos]  # (batch, d_model)
            token_vec = self.token_vectors[pos]  # (d_model,)

            # Dot product gives token contribution
            contrib = einsum(grad_at_pos, token_vec, "batch d, d -> batch")

            token_idx = self.n_features + pos
            self._batch_buffer[token_idx] = contrib

    def _zero_grads(self):
        """Zero all gradients before a new backward pass."""
        # Zero feature gradients
        for layer in self.model.model.model.layers:
            if layer.mlp.cached_features is not None:
                if layer.mlp.cached_features.grad is not None:
                    layer.mlp.cached_features.grad.zero_()

        # Zero embedding gradients
        embed_resid = self._resid_cache.get('embed')
        if embed_resid is not None and embed_resid.grad is not None:
            embed_resid.grad.zero_()

        # Zero all cached residual gradients
        for key, resid in self._resid_cache.items():
            if resid is not None and resid.grad is not None:
                resid.grad.zero_()

    def clear_cache(self):
        """Clear cached activations."""
        self._resid_cache.clear()

        # Also clear cached features in model
        for layer in self.model.model.model.layers:
            layer.mlp.cached_features = None
