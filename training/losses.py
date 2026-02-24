"""Loss functions for bridging experiments."""

import torch
import torch.nn.functional as F


def compute_kl_loss(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute KL divergence from reference logits for next-token prediction.

    KL(ref || model) - we want model to match ref distribution.
    Shifts logits by 1 to match LM loss convention (predicting next token).

    Args:
        logits: model logits [batch, seq_len, vocab_size]
        ref_logits: reference model logits [batch, seq_len, vocab_size]
        labels: optional labels to mask padding [batch, seq_len]
        ignore_index: index to ignore in labels

    Returns:
        Scalar KL divergence loss
    """
    # Ensure tensors are on the same device (for multi-GPU)
    device = logits.device
    ref_logits = ref_logits.to(device)
    if labels is not None:
        labels = labels.to(device)

    # Shift for next-token prediction (same as LM loss)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()

    # Flatten to [batch * (seq_len-1), vocab_size]
    logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    ref_logits_flat = shift_ref_logits.view(-1, shift_ref_logits.size(-1))

    # Compute log probs and probs
    log_probs = F.log_softmax(logits_flat, dim=-1)
    ref_probs = F.softmax(ref_logits_flat, dim=-1)

    # KL divergence: sum over vocab, mean over tokens
    kl = F.kl_div(log_probs, ref_probs, reduction='none').sum(dim=-1)

    # Mask out padding if labels provided (use shifted labels)
    if labels is not None:
        shift_labels = labels[..., 1:].contiguous()
        mask = (shift_labels.view(-1) != ignore_index).float()
        kl = (kl * mask).sum() / mask.sum().clamp(min=1)
    else:
        kl = kl.mean()

    return kl


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute standard language modeling cross-entropy loss.

    Args:
        logits: model logits [batch, seq_len, vocab_size]
        labels: target token ids [batch, seq_len]
        ignore_index: index to ignore in loss computation

    Returns:
        Scalar cross-entropy loss
    """
    # Ensure tensors are on the same device (for multi-GPU)
    labels = labels.to(logits.device)

    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten and compute loss
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    return loss


def _layer_nmse(h_adapt: torch.Tensor, h_ref: torch.Tensor) -> torch.Tensor:
    """Compute NMSE for a single layer's hidden states."""
    # Ensure both tensors are on the same device
    h_ref = h_ref.to(h_adapt.device)
    mse = (h_adapt - h_ref).pow(2).mean()
    norm = h_ref.pow(2).mean()
    return mse / (norm + 1e-8)


def compute_nmse_loss(
    model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    return_layerwise: bool = False,
):
    """
    Compute NMSE between hidden states at all layers, memory-efficiently.

    Runs both models layer-by-layer, computing NMSE incrementally without
    storing all hidden states. Only 2 hidden states are kept at a time.

    Args:
        model: adapter model (gradients enabled)
        ref_model: reference model (frozen)
        input_ids: input token ids [batch, seq_len]
        attention_mask: attention mask [batch, seq_len]
        return_layerwise: if True, also return dict of per-layer NMSE values

    Returns:
        If return_layerwise=False: Scalar NMSE loss averaged over all layers
        If return_layerwise=True: (scalar_loss, dict of layer -> nmse_value)
    """
    from transformers.masking_utils import create_causal_mask

    n_layers = len(model.model.layers)
    device = model.get_input_embeddings().weight.device

    # Get embeddings
    h_adapt = model.model.embed_tokens(input_ids)
    with torch.no_grad():
        h_ref = ref_model.model.embed_tokens(input_ids)

    # Setup position ids and cache position
    seq_len = h_adapt.shape[1]
    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0)

    # Create causal mask
    mask_kwargs = {
        "config": model.config,
        "input_embeds": h_adapt,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask = create_causal_mask(**mask_kwargs) # type: ignore

    # Position embeddings
    position_embeddings_adapt = model.model.rotary_emb(h_adapt, position_ids)
    with torch.no_grad():
        position_embeddings_ref = ref_model.model.rotary_emb(h_ref, position_ids)

    # NMSE on embeddings (layer 0)
    layer_nmse_0 = _layer_nmse(h_adapt, h_ref.detach())
    nmse_total = layer_nmse_0.to(device)
    layerwise = {0: layer_nmse_0.item()} if return_layerwise else None

    # Layer by layer
    for i, (layer_adapt, layer_ref) in enumerate(zip(model.model.layers, ref_model.model.layers)):
        layer_mask = causal_mask  # Assumes full attention for all layers

        h_adapt = layer_adapt(
            h_adapt,
            attention_mask=layer_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings_adapt,
            cache_position=cache_position,
        )

        with torch.no_grad():
            h_ref = layer_ref(
                h_ref,
                attention_mask=layer_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings_ref,
                cache_position=cache_position,
            )

        # Accumulate NMSE (move to consistent device for multi-GPU)
        layer_nmse_i = _layer_nmse(h_adapt, h_ref.detach())
        nmse_total = nmse_total + layer_nmse_i.to(device)
        if return_layerwise:
            assert layerwise is not None
            layerwise[i + 1] = layer_nmse_i.item()

    # Average over all layers (embeddings + n_layers)
    avg_nmse = nmse_total / (n_layers + 1)

    if return_layerwise:
        return avg_nmse, layerwise
    return avg_nmse
