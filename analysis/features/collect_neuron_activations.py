"""
Collect MLP neuron activations for visualization (baseline comparison).

Collects activating examples for MLP neurons in standard transformer models
(e.g., DeepSeek R1 Distill) for comparison with transcoder features.
Outputs per-neuron JSONs in the same circuit-tracer format as feature collection.

Usage:
    python -m analysis.features.collect_neuron_activations \
        --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --val_data hf://nathu0/transcoder-adapters-openthoughts3-stratified-55k/data/val.jsonl \
        --output_dir ./neuron_data

Key differences from collect_feature_activations.py:
- Captures MLP intermediate activations: act(gate_proj(x)) * up_proj(x)
- Samples a subset of neurons per layer (default 500)
- Uses down_proj for logit lens instead of transcoder_dec
"""

from pathlib import Path

import argparse
import json
import random
import heapq
from collections import defaultdict
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from training.dataset import OpenThoughtsDataset

# Reuse data structures and helpers from the feature collection script
from analysis.features.collect_feature_activations import (
    ActivatingExample,
    FeatureStats,
    get_special_token_ids,
    find_token_positions,
    precompute_regions,
    cantor_pair,
    format_example_for_circuit_tracer,
    _write_feature_json,
)


# =============================================================================
# Neuron Collector (modified from FeatureCollector)
# =============================================================================

class NeuronCollector:
    """Collects neuron activation statistics across sequences."""

    def __init__(
        self,
        n_layers: int,
        intermediate_size: int,
        sampled_neurons: list[list[int]],  # [layer][neuron_indices]
        top_k: int = 20,
        n_random: int = 10,
        context_before: int = 50,
        context_after: int = 20,
    ):
        self.n_layers = n_layers
        self.intermediate_size = intermediate_size
        self.sampled_neurons = sampled_neurons  # Which neurons to track per layer
        self.top_k = top_k
        self.n_random = n_random
        self.context_before = context_before
        self.context_after = context_after

        # Create index mapping: sampled_neurons[layer] -> list of neuron indices
        # stats[layer][local_idx] corresponds to neuron sampled_neurons[layer][local_idx]
        self.stats = [
            [FeatureStats(layer_idx=l, feature_idx=sampled_neurons[l][i])
             for i in range(len(sampled_neurons[l]))]
            for l in range(n_layers)
        ]

        # Reverse mapping: (layer, neuron_idx) -> local_idx in stats
        self.neuron_to_local = [
            {neuron_idx: local_idx for local_idx, neuron_idx in enumerate(sampled_neurons[l])}
            for l in range(n_layers)
        ]

        # Pre-computed tensors for GPU indexing (created on first use)
        self._sampled_tensors: dict[int, torch.Tensor] = {}

        # Global counts for normalization
        self.total_tokens = 0
        self.tokens_per_domain: dict[str, int] = defaultdict(int)
        self.tokens_per_region: dict[str, int] = defaultdict(int)
        self.tokens_per_thinking_bin: list[int] = [0] * 10

        # Hook storage
        self._hooks: list = []
        self._layer_activations: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        """Create a forward hook that captures MLP intermediate activations."""
        def hook(module, input, output):
            hidden_states = input[0]  # [batch, seq, d_model]
            with torch.no_grad():
                # Qwen2 SwiGLU: act(gate_proj(x)) * up_proj(x)
                gate = module.act_fn(module.gate_proj(hidden_states))
                up = module.up_proj(hidden_states)
                neurons = gate * up  # [batch, seq, intermediate_size]
            self._layer_activations[layer_idx] = neurons[0]  # [seq, intermediate_size]
        return hook

    def register_hooks(self, model):
        """Register forward hooks on all MLP layers."""
        self._hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            hook = layer.mlp.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _get_context(self, tokens: list[int], position: int) -> tuple[list[int], int]:
        """Extract context window around a position."""
        start = max(0, position - self.context_before)
        end = min(len(tokens), position + self.context_after + 1)
        return tokens[start:end], position - start

    def _maybe_add_example(
        self,
        stats: FeatureStats,
        activation: float,
        tokens: list[int],
        position: int,
        neurons_gpu: torch.Tensor,
        neuron_idx: int,
        domain: str,
        region: str,
        thinking_position: float | None,
        sequence_idx: int,
    ):
        """Add example to top-k heap if it qualifies."""
        # Check if this could make it into top-k
        if (len(stats.top_k_examples) >= self.top_k
                and activation <= stats.top_k_examples[0].activation):
            return

        # Create example
        context_tokens, pos_in_ctx = self._get_context(tokens, position)
        ctx_start = max(0, position - self.context_before)
        ctx_end = min(len(tokens), position + self.context_after + 1)
        context_activations = neurons_gpu[ctx_start:ctx_end, neuron_idx].float().cpu().tolist()

        example = ActivatingExample(
            activation=activation,
            token_id=tokens[position],
            position=position,
            context_tokens=context_tokens,
            context_activations=context_activations,
            position_in_context=pos_in_ctx,
            domain=domain,
            region=region,
            thinking_position=thinking_position,
            sequence_idx=sequence_idx,
        )

        # Add to top-k heap
        if len(stats.top_k_examples) < self.top_k:
            heapq.heappush(stats.top_k_examples, example)
        else:
            heapq.heapreplace(stats.top_k_examples, example)

    def _get_sampled_tensor(self, layer_idx: int, device) -> torch.Tensor:
        """Get cached sampled neuron indices tensor for a layer."""
        if layer_idx not in self._sampled_tensors:
            self._sampled_tensors[layer_idx] = torch.tensor(
                self.sampled_neurons[layer_idx], device=device, dtype=torch.long
            )
        return self._sampled_tensors[layer_idx]

    def process_sequence(
        self,
        model,
        tokens: list[int],
        domain: str,
        markers: dict,
        sequence_idx: int,
    ):
        """Process a single sequence and update neuron stats."""
        seq_len = len(tokens)
        self._layer_activations = {}

        # Forward pass (hooks capture activations)
        input_ids = torch.tensor([tokens], device=model.device)
        with torch.no_grad():
            model(input_ids)

        # Precompute regions for all positions
        regions, thinking_positions = precompute_regions(tokens, markers)

        # Update global token counts
        self.total_tokens += seq_len
        self.tokens_per_domain[domain] += seq_len
        for pos in range(seq_len):
            self.tokens_per_region[regions[pos]] += 1
            think_pos_val = thinking_positions[pos]
            if think_pos_val is not None:
                bin_idx = min(9, int(think_pos_val * 10))
                self.tokens_per_thinking_bin[bin_idx] += 1

        # Process each layer using topk (much faster than iterating all activations)
        for layer_idx in range(self.n_layers):
            neurons_gpu = self._layer_activations[layer_idx]  # [seq_len, intermediate_size]
            sampled = self.sampled_neurons[layer_idx]
            n_sampled = len(sampled)

            # Get sampled neurons (cached tensor)
            sampled_tensor = self._get_sampled_tensor(layer_idx, neurons_gpu.device)
            sampled_activations = neurons_gpu[:, sampled_tensor]  # [seq_len, n_sampled]

            # Get top-k positions per neuron using topk (vectorized, no Python loop)
            k = min(self.top_k, seq_len)
            top_vals, top_positions = sampled_activations.T.topk(k, dim=1)  # [n_sampled, k]

            # Transfer to CPU once
            top_vals_cpu = top_vals.float().cpu().numpy()  # [n_sampled, k]
            top_positions_cpu = top_positions.cpu().numpy()  # [n_sampled, k]

            # Process top-k for each neuron (small loop: n_sampled * k iterations)
            for local_idx in range(n_sampled):
                neuron_idx = sampled[local_idx]
                stats = self.stats[layer_idx][local_idx]

                for j in range(k):
                    pos = int(top_positions_cpu[local_idx, j])
                    act = float(top_vals_cpu[local_idx, j])

                    if act <= 0.01:  # Skip weak activations
                        continue

                    region = regions[pos]
                    think_pos = thinking_positions[pos]

                    # Update counts (approximate - only counts top activations)
                    stats.activation_count += 1
                    stats.domain_counts[domain] += 1
                    stats.region_counts[region] += 1
                    if think_pos is not None:
                        bin_idx = min(9, int(think_pos * 10))
                        stats.thinking_position_counts[bin_idx] += 1

                    # Maybe add to examples
                    self._maybe_add_example(
                        stats, act, tokens, pos, neurons_gpu, neuron_idx,
                        domain, region, think_pos, sequence_idx
                    )

        # Clear activations
        self._layer_activations = {}


# =============================================================================
# Logit Lens for Neurons
# =============================================================================

def compute_logit_lens_neurons(
    model,
    tokenizer,
    sampled_neurons: list[list[int]],
    top_k: int = 10,
) -> list[dict]:
    """
    Compute top/bottom unembed tokens for sampled neurons.

    Uses down_proj weights to find which tokens each neuron promotes/suppresses.
    """
    print("Computing logit lens for neurons...")

    unembed = model.lm_head.weight.data  # [vocab_size, d_model]

    logit_lens_data = []

    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc="Logit lens")):
        mlp = layer.mlp
        # down_proj: [d_model, intermediate_size]
        down_proj = mlp.down_proj.weight.data

        sampled = sampled_neurons[layer_idx]
        n_sampled = len(sampled)

        # Get columns for sampled neurons only
        sampled_down = down_proj[:, sampled]  # [d_model, n_sampled]

        with torch.no_grad():
            logits = unembed @ sampled_down  # [vocab_size, n_sampled]

        top_vals, top_ids = logits.topk(top_k, dim=0)
        bot_vals, bot_ids = logits.topk(top_k, dim=0, largest=False)

        layer_data = {
            'neuron_indices': sampled,
            'top_ids': top_ids.T.cpu().tolist(),
            'top_vals': top_vals.T.cpu().tolist(),
            'bot_ids': bot_ids.T.cpu().tolist(),
            'bot_vals': bot_vals.T.cpu().tolist(),
        }
        logit_lens_data.append(layer_data)

    return logit_lens_data


# =============================================================================
# Export Functions
# =============================================================================

def export_circuit_tracer_json_neurons(
    collector: NeuronCollector,
    logit_lens_data: list[dict],
    tokenizer,
    output_dir: Path,
    n_workers: int = 16,
):
    """Export neuron data to circuit tracer JSON format."""
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting neurons to {features_dir}...")

    write_tasks = []
    skipped = 0

    for layer_idx in tqdm(range(collector.n_layers), desc="Building JSON"):
        layer_logit_lens = logit_lens_data[layer_idx]
        sampled = collector.sampled_neurons[layer_idx]

        for local_idx, neuron_idx in enumerate(sampled):
            stats = collector.stats[layer_idx][local_idx]

            if stats.activation_count == 0:
                skipped += 1
                continue

            top_examples = sorted(stats.top_k_examples, key=lambda x: -x.activation)
            random_examples = stats.random_examples

            top_formatted = [format_example_for_circuit_tracer(ex, tokenizer) for ex in top_examples]
            random_formatted = [format_example_for_circuit_tracer(ex, tokenizer) for ex in random_examples]

            all_acts = []
            for ex in top_examples + random_examples:
                all_acts.extend(ex.context_activations)
            act_min = min(all_acts) if all_acts else 0.0
            act_max = max(all_acts) if all_acts else 1.0

            top_logits = [tokenizer.decode([tok_id]) for tok_id in layer_logit_lens['top_ids'][local_idx]]
            bottom_logits = [tokenizer.decode([tok_id]) for tok_id in layer_logit_lens['bot_ids'][local_idx]]

            feature_json = {
                "top_logits": top_logits,
                "bottom_logits": bottom_logits,
                "act_min": act_min,
                "act_max": act_max,
                "examples_quantiles": [
                    {"quantile_name": "Top activations", "examples": top_formatted},
                    {"quantile_name": "Random samples", "examples": random_formatted},
                ],
                "activation_frequency": stats.activation_count / max(1, collector.total_tokens),
                "layer": layer_idx,
                "feature": neuron_idx,  # Original neuron index
                "is_neuron": True,  # Mark as neuron baseline
            }

            cantor_id = cantor_pair(layer_idx, neuron_idx)
            filepath = features_dir / f"{cantor_id}.json"
            write_tasks.append((filepath, feature_json))

    print(f"Writing {len(write_tasks)} files with {n_workers} workers...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(executor.map(_write_feature_json, write_tasks), total=len(write_tasks), desc="Writing files"))

    print(f"Generated {len(write_tasks)} neuron files, skipped {skipped} empty neurons")


def export_metadata_neurons(collector: NeuronCollector, output_dir: Path):
    """Export metadata for neurons."""
    print("Exporting metadata...")

    metadata: dict[str, Any] = {
        "total_tokens": collector.total_tokens,
        "tokens_per_domain": dict(collector.tokens_per_domain),
        "tokens_per_region": dict(collector.tokens_per_region),
        "tokens_per_thinking_bin": collector.tokens_per_thinking_bin,
        "is_neuron_baseline": True,
        "features": [],
    }

    for layer_idx in range(collector.n_layers):
        for local_idx, neuron_idx in enumerate(collector.sampled_neurons[layer_idx]):
            stats = collector.stats[layer_idx][local_idx]

            if stats.activation_count == 0:
                continue

            total_acts = stats.activation_count

            domain_density = {}
            domain_fraction = {}
            for domain, count in stats.domain_counts.items():
                domain_tokens = collector.tokens_per_domain.get(domain, 0)
                domain_density[domain] = count / domain_tokens if domain_tokens > 0 else 0
                domain_fraction[domain] = count / total_acts

            region_density = {}
            region_fraction = {}
            for region, count in stats.region_counts.items():
                region_tokens = collector.tokens_per_region.get(region, 0)
                region_density[region] = count / region_tokens if region_tokens > 0 else 0
                region_fraction[region] = count / total_acts

            thinking_density = []
            thinking_fraction = []
            thinking_total = sum(stats.thinking_position_counts)
            for bin_idx in range(10):
                bin_acts = stats.thinking_position_counts[bin_idx]
                bin_tokens = collector.tokens_per_thinking_bin[bin_idx]
                thinking_density.append(bin_acts / bin_tokens if bin_tokens > 0 else 0)
                thinking_fraction.append(bin_acts / thinking_total if thinking_total > 0 else 0)

            feature_meta = {
                "layer": layer_idx,
                "feature": neuron_idx,
                "cantor_id": cantor_pair(layer_idx, neuron_idx),
                "activation_count": total_acts,
                "activation_freq": total_acts / collector.total_tokens,
                "domain_density": domain_density,
                "domain_fraction": domain_fraction,
                "region_density": region_density,
                "region_fraction": region_fraction,
                "thinking_position_density": thinking_density,
                "thinking_position_fraction": thinking_fraction,
                "is_neuron": True,
            }
            metadata["features"].append(feature_meta)

    with open(output_dir / "feature_metadata.json", 'w') as f:
        json.dump(metadata, f)

    print(f"Saved metadata for {len(metadata['features'])} neurons")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect MLP neuron activations for visualization (baseline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF repo ID or local path to model")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Path to validation JSONL (local or hf://)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for neuron JSONs and metadata")

    # Neuron sampling
    parser.add_argument("--n_neurons_per_layer", type=int, default=500,
                        help="Number of neurons to sample per layer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for neuron sampling")

    # Other args (same as transcoder version)
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Tokenizer (default: DeepSeek R1 Distill for data compatibility)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top activating examples per neuron")
    parser.add_argument("--n_random", type=int, default=10,
                        help="Number of random samples per neuron")
    parser.add_argument("--context_before", type=int, default=50,
                        help="Context tokens before activating token")
    parser.add_argument("--context_after", type=int, default=20,
                        help="Context tokens after activating token")
    parser.add_argument("--max_length", type=int, default=10000,
                        help="Max sequence length")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Tokenizer (always use DeepSeek for data compatibility)
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Model
    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = len(model.model.layers)
    intermediate_size = model.config.intermediate_size
    print(f"Model: {n_layers} layers, {intermediate_size} intermediate size")

    # Sample neurons
    random.seed(args.seed)
    sampled_neurons = []
    for layer_idx in range(n_layers):
        n_sample = min(args.n_neurons_per_layer, intermediate_size)
        sampled = sorted(random.sample(range(intermediate_size), n_sample))
        sampled_neurons.append(sampled)
    print(f"Sampled {args.n_neurons_per_layer} neurons per layer ({n_layers * args.n_neurons_per_layer} total)")

    # Dataset
    print(f"Loading validation data: {args.val_data}")
    dataset = OpenThoughtsDataset(
        data_path=args.val_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        format="deepseek",
        truncate=True,
        loss_on_prompt=False,
    )

    n_samples = len(dataset)
    if args.max_samples:
        n_samples = min(args.max_samples, n_samples)
    print(f"Processing {n_samples} samples")

    # Collector
    collector = NeuronCollector(
        n_layers=n_layers,
        intermediate_size=intermediate_size,
        sampled_neurons=sampled_neurons,
        top_k=args.top_k,
        n_random=args.n_random,
        context_before=args.context_before,
        context_after=args.context_after,
    )

    collector.register_hooks(model)

    # Process
    skipped = 0
    for idx in tqdm(range(n_samples), desc="Processing sequences"):
        item = dataset[idx]
        meta = dataset.examples[idx]

        tokens = item['input_ids']
        domain = meta.get('domain', 'unknown')

        markers = find_token_positions(tokens, tokenizer)

        if markers['think_start'] is None or markers['think_end'] is None:
            skipped += 1
            continue

        collector.process_sequence(model, tokens, domain, markers, idx)

    collector.remove_hooks()

    if skipped > 0:
        print(f"Skipped {skipped} samples due to missing <think> tags")

    print(f"\nCollection summary:")
    print(f"  Total tokens: {collector.total_tokens:,}")
    print(f"  Domains: {dict(collector.tokens_per_domain)}")
    print(f"  Regions: {dict(collector.tokens_per_region)}")

    # Logit lens
    logit_lens_data = compute_logit_lens_neurons(model, tokenizer, sampled_neurons)

    # Export
    export_circuit_tracer_json_neurons(collector, logit_lens_data, tokenizer, output_dir)
    export_metadata_neurons(collector, output_dir)

    print(f"\nDone! Output written to {output_dir}")


if __name__ == "__main__":
    main()
