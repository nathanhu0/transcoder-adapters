"""
RelP attribution: compute edge weights between features, tokens, and logits.

Produces a Graph compatible with upstream circuit-tracer's prune_graph and
create_graph_files by padding the adjacency matrix with zero error nodes.
"""

import logging
import time

import torch
from tqdm import tqdm

from circuit_tracer.graph import Graph


@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_proj: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the smallest logit set whose cumulative prob >= *desired_logit_prob*.

    Args:
        logits: ``(d_vocab,)`` vector (single position).
        unembed_proj: ``(d_model, d_vocab)`` unembedding matrix.
        max_n_logits: Hard cap *k*.
        desired_logit_prob: Cumulative probability threshold *p*.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            * logit_indices - ``(k,)`` vocabulary ids.
            * logit_probs   - ``(k,)`` softmax probabilities.
            * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned.
    """

    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    cols = unembed_proj[:, top_idx]
    demeaned = cols - unembed_proj.mean(dim=-1, keepdim=True)
    return top_idx, top_p, demeaned.T


def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None):
    """Compute partial influences using power iteration method.

    Vendored from circuit_tracer.graph for clarity — could also be imported directly.
    """
    device = device or edge_matrix.device

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p) :] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index] @ normalized_matrix
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences



def attribute(
    prompt,
    model,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    verbose: bool = False,
    update_interval: int = 4,
) -> Graph:
    """Compute an attribution graph for a prompt using RelP backward.

    Args:
        prompt: Text, token ids, or tensor.
        model: RelPReplacementModel instance.
        max_n_logits: Max number of logit nodes.
        desired_logit_prob: Keep logits until cumulative prob >= this value.
        batch_size: Source nodes per backward pass.
        max_feature_nodes: Cap on feature nodes (None = all).
        verbose: Show progress.
        update_interval: Batches between influence re-ranking.

    Returns:
        Graph compatible with upstream circuit-tracer visualization.
    """
    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if verbose and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    try:
        return _run_relp_attribution(
            model=model,
            prompt=prompt,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            verbose=verbose,
            update_interval=update_interval,
            logger=logger,
        )
    finally:
        if handler:
            logger.removeHandler(handler)


def _run_relp_attribution(
    model,
    prompt,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    verbose,
    update_interval,
    logger,
):
    """Internal: run RelP attribution with feature.grad collection."""
    start_time = time.time()

    # =========================================================================
    # Phase 0: Setup
    # =========================================================================
    logger.info("Phase 0: Setting up attribution context")
    phase_start = time.time()

    ctx = model.setup_attribution(prompt)
    input_ids = model.ensure_tokenized(prompt)
    activation_matrix = ctx.activation_matrix

    n_layers = model.cfg.n_layers
    n_pos = len(input_ids)
    n_features = activation_matrix._nnz()

    logger.info(f"Setup completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {n_features} active features across {n_layers} layers, {n_pos} positions")

    # =========================================================================
    # Phase 1: Compute salient logits
    # =========================================================================
    logger.info("Phase 1: Computing salient logits")
    phase_start = time.time()

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        ctx.logits[0, -1],
        model.unembed.W_U,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )
    n_logits = len(logit_idx)
    logger.info(f"Selected {n_logits} logits with cumulative prob {logit_p.sum().item():.4f}")
    logger.info(f"Phase 1 completed in {time.time() - phase_start:.2f}s")

    # =========================================================================
    # Phase 2: Setup edge matrix
    # =========================================================================
    logger.info("Phase 2: Setting up edge matrix")
    phase_start = time.time()

    # Node layout: [features | tokens | logits]
    # - features: n_features nodes (have incoming edges)
    # - tokens: n_pos nodes (source only)
    # - logits: n_logits nodes (sink)
    logit_offset = n_features + n_pos
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or n_features, n_features)
    logger.info(f"Will include up to {max_feature_nodes} of {n_features} feature nodes")

    # Edge matrix: rows are [logits | features], cols are [features | tokens | logits]
    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)

    logger.info(f"Phase 2 completed in {time.time() - phase_start:.2f}s")

    # =========================================================================
    # Phase 3: Logit attribution
    # =========================================================================
    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()

    feat_layers, feat_pos, _ = activation_matrix.indices()

    # Run forward pass to setup caches
    ctx._run_forward_with_cache(batch_size)

    for i in range(0, n_logits, batch_size):
        batch_end = min(i + batch_size, n_logits)
        batch_vecs = logit_vecs[i:batch_end]
        actual_batch = batch_vecs.shape[0]

        rows = ctx.compute_batch(
            layers=torch.full((actual_batch,), n_layers, device=model.cfg.device),
            positions=torch.full((actual_batch,), n_pos - 1, device=model.cfg.device),
            inject_values=batch_vecs,
            retain_graph=True,
        )

        edge_matrix[i:batch_end, :logit_offset] = rows[:, :logit_offset].cpu()
        row_to_node_index[i:batch_end] = torch.arange(i, batch_end) + logit_offset

    logger.info(f"Phase 3 completed in {time.time() - phase_start:.2f}s")

    # =========================================================================
    # Phase 4: Feature attribution (influence-ranked)
    # =========================================================================
    logger.info("Phase 4: Computing feature attributions")
    phase_start = time.time()

    st = n_logits  # Start writing feature rows after logit rows
    visited = torch.zeros(n_features, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature attribution", disable=not verbose)

    while n_visited < max_feature_nodes:
        # Compute influence ranking
        if max_feature_nodes == n_features:
            # Process all - just take unvisited in order
            pending = torch.where(~visited)[0][:batch_size]
        else:
            # Rank by partial influence
            influences = compute_partial_influences(
                edge_matrix[:st], logit_p, row_to_node_index[:st]
            )
            feature_rank = torch.argsort(influences[:n_features], descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        if len(pending) == 0:
            break

        # Process in batches
        for batch_start in range(0, len(pending), batch_size):
            idx_batch = pending[batch_start:batch_start + batch_size]
            actual_batch = len(idx_batch)

            n_visited += actual_batch

            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=ctx.encoder_vecs[idx_batch],
                retain_graph=n_visited < max_feature_nodes,
            )

            end = st + actual_batch
            edge_matrix[st:end, :logit_offset] = rows[:, :logit_offset].cpu()
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end

            pbar.update(actual_batch)

    pbar.close()
    logger.info(f"Phase 4 completed in {time.time() - phase_start:.2f}s")

    # =========================================================================
    # Phase 5: Build graph with error padding
    # =========================================================================
    logger.info("Phase 5: Building graph")
    phase_start = time.time()

    selected_features = torch.where(visited)[0]
    n_selected = len(selected_features)

    # Remap columns if we didn't select all features
    if n_selected < n_features:
        col_read_features = selected_features
        col_read_tokens = torch.arange(n_features, n_features + n_pos)
        col_read_logits = torch.arange(logit_offset, total_nodes)
        col_read = torch.cat([col_read_features, col_read_tokens, col_read_logits])
        edge_matrix = edge_matrix[:, col_read]

    # Sort rows by node index
    edge_matrix = edge_matrix[row_to_node_index.argsort()]

    # Build adjacency matrix with layout: [features | tokens | logits]
    compact_count = n_selected + n_pos + n_logits
    compact_matrix = torch.zeros(compact_count, compact_count)
    compact_matrix[:n_selected] = edge_matrix[:n_selected]
    compact_matrix[-n_logits:] = edge_matrix[n_selected:n_selected + n_logits]

    # Zero BOS (position 0) — features at pos 0 and token 0 should not contribute
    for i in range(n_selected):
        feat_pos_i = activation_matrix.indices()[1][selected_features[i]].item()
        if feat_pos_i == 0:
            compact_matrix[:, i] = 0
            compact_matrix[i, :] = 0
    compact_matrix[:, n_selected] = 0  # token position 0 column
    compact_matrix[n_selected, :] = 0  # token position 0 row (should already be 0)

    # Pad with n_layers * n_pos zero error rows/cols between features and tokens
    # Upstream Graph expects: [features | error | tokens | logits]
    n_error = n_layers * n_pos
    final_count = n_selected + n_error + n_pos + n_logits
    full_edge_matrix = torch.zeros(final_count, final_count)

    # Copy feature block (rows and cols 0..n_selected-1)
    full_edge_matrix[:n_selected, :n_selected] = compact_matrix[:n_selected, :n_selected]
    # Copy feature->token edges (shift token cols by n_error)
    full_edge_matrix[:n_selected, n_selected + n_error:] = compact_matrix[:n_selected, n_selected:]
    # Copy logit->feature edges
    full_edge_matrix[-n_logits:, :n_selected] = compact_matrix[-n_logits:, :n_selected]
    # Copy logit->token edges (shift token cols by n_error)
    full_edge_matrix[-n_logits:, n_selected + n_error:] = compact_matrix[-n_logits:, n_selected:]

    logger.info(f"Phase 5 completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Adjacency: {n_selected} features + {n_error} error(zeros) + {n_pos} tokens + {n_logits} logits")

    # Build graph object — pass HF config for upstream convert_nnsight_config_to_transformerlens
    # Inject head_dim if missing (Qwen2 config computes it but doesn't store it explicitly)
    cfg = model.hf_config
    if not hasattr(cfg, 'head_dim') or cfg.head_dim is None:
        cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=cfg,
        scan=model.scan,
    )

    total_time = time.time() - start_time
    logger.info(f"RelP attribution completed in {total_time:.2f}s")

    return graph
