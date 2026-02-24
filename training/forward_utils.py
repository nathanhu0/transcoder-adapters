"""Forward pass utilities for bridging experiments."""

import torch
import random

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


def forward_mixed(
    model1,
    model2,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    switch_layer: int,
) -> torch.Tensor:
    """
    Forward pass that switches from model1 to model2 at switch_layer.

    Runs model1 for layers 0..switch_layer-1, then model2 for layers switch_layer..L.
    Uses model1's embeddings and model2's final norm + lm_head.

    Args:
        model1: First model (e.g., adapter model)
        model2: Second model (e.g., reference model)
        input_ids: input token ids [batch, seq_len]
        attention_mask: attention mask [batch, seq_len]
        switch_layer: layer index to switch at (0 = all model2, L = all model1)

    Returns:
        logits: output logits [batch, seq_len, vocab_size]
    """
    # Get embeddings from model1
    h = model1.model.embed_tokens(input_ids)

    # Setup position ids and cache position
    seq_len = h.shape[1]
    device = h.device
    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0)

    # Create causal mask (use model1's config, should be same arch)
    mask_kwargs = {
        "config": model1.config,
        "input_embeds": h,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs), # type: ignore
    }
    if any(getattr(layer, 'attention_type', None) == 'sliding_attention' for layer in model1.model.layers):
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs) # type: ignore

    # Position embeddings from model1
    position_embeddings = model1.model.rotary_emb(h, position_ids)

    # Model1 layers: 0 to switch_layer-1
    for layer in model1.model.layers[:switch_layer]:
        layer_attn_mask = causal_mask_mapping[layer.attention_type]
        h = layer(
            h,
            attention_mask=layer_attn_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )

    # Model2 layers: switch_layer to L
    # Need model2's position embeddings for its layers
    position_embeddings_2 = model2.model.rotary_emb(h, position_ids)
    for layer in model2.model.layers[switch_layer:]:
        layer_attn_mask = causal_mask_mapping[layer.attention_type]
        h = layer(
            h,
            attention_mask=layer_attn_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings_2,
            cache_position=cache_position,
        )

    # Final norm + lm_head from model2
    h = model2.model.norm(h)
    logits = model2.lm_head(h)

    return logits


def sample_cutoffs(
    n_layers: int,
    n_cutoffs: int,
    sampling: str | list[int] = "uniform",
) -> list[int]:
    """
    Sample layer cutoff indices without replacement.

    Args:
        n_layers: total number of layers in the model
        n_cutoffs: number of cutoffs to sample
        sampling: "uniform" or list of fixed layer indices

    Returns:
        List of layer indices in [0, n_layers] (inclusive)
        - 0 means switch at the very beginning (no layers from first model)
        - n_layers means switch at the very end (all layers from first model)
    """
    if isinstance(sampling, list):
        # Fixed layer indices
        return sampling

    if sampling == "uniform":
        # Sample without replacement from [0, n_layers]
        all_layers = list(range(n_layers + 1))
        n_to_sample = min(n_cutoffs, len(all_layers))
        return sorted(random.sample(all_layers, n_to_sample))

    raise ValueError(f"Unknown sampling strategy: {sampling}")
