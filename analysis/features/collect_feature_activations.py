"""
Collect transcoder feature activations for visualization.

Runs the model on validation data and collects top-activating examples,
logit lens, and activation statistics for each transcoder feature.
Outputs per-feature JSONs compatible with the circuit-tracer frontend.

Usage:
    python -m analysis.features.collect_feature_activations \
        --model_path nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4 \
        --val_data hf://nathu0/transcoder-adapters-openthoughts3-stratified-55k/data/val.jsonl \
        --output_dir ./feature_data

Output:
    {output_dir}/
    ├── features/               # Per-feature JSON files (circuit-tracer format)
    │   ├── {cantor_id}.json   # cantor_pair(layer, feature) -> unique int
    │   └── ...
    └── feature_metadata.json  # Activation frequencies, domain/region breakdowns
"""

from pathlib import Path

import argparse
import json
import random
import heapq
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from training.dataset import OpenThoughtsDataset
from models.qwen2_transcoder import Qwen2ForCausalLMWithTranscoder


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ActivatingExample:
    """A single activating example for a feature."""
    activation: float
    token_id: int
    position: int
    context_tokens: list[int]
    context_activations: list[float]
    position_in_context: int
    # Metadata
    domain: str
    region: str  # bos, user_marker, question, assistant_marker, think_start, thinking, think_end, answer
    thinking_position: Optional[float]  # 0-1 if in thinking region, else None
    sequence_idx: int

    def __lt__(self, other):
        """For heap comparison (min-heap on activation)."""
        return self.activation < other.activation


@dataclass
class FeatureStats:
    """Stats for a single feature."""
    layer_idx: int
    feature_idx: int

    # Examples (min-heap for top-k, list for reservoir sampling)
    top_k_examples: list = field(default_factory=list)
    random_examples: list = field(default_factory=list)
    random_seen_count: int = 0  # for reservoir sampling

    # Activation count
    activation_count: int = 0

    # Per-domain activation counts
    domain_counts: dict = field(default_factory=lambda: defaultdict(int))

    # Per-region activation counts
    region_counts: dict = field(default_factory=lambda: defaultdict(int))

    # Thinking position histogram (10 bins)
    thinking_position_counts: list = field(default_factory=lambda: [0] * 10)


# =============================================================================
# Region Classification
# =============================================================================

# Cache for special token IDs (populated on first use)
_SPECIAL_TOKEN_IDS = {}


def get_special_token_ids(tokenizer) -> dict:
    """Get token IDs for special tokens (cached)."""
    global _SPECIAL_TOKEN_IDS
    if not _SPECIAL_TOKEN_IDS:
        # These are the DeepSeek R1 special tokens
        _SPECIAL_TOKEN_IDS = {
            'bos': tokenizer.bos_token_id,
            'user_marker': tokenizer.encode("<｜User｜>", add_special_tokens=False)[0],
            'assistant_marker': tokenizer.encode("<｜Assistant｜>", add_special_tokens=False)[0],
            'think_start': tokenizer.encode("<think>", add_special_tokens=False)[0],
            'think_end': tokenizer.encode("</think>", add_special_tokens=False)[0],
        }
    return _SPECIAL_TOKEN_IDS


def find_token_positions(tokens: list[int], tokenizer) -> dict:
    """Find positions of special tokens in a sequence."""
    special_ids = get_special_token_ids(tokenizer)

    positions = {
        'bos': None,
        'user_marker': None,
        'assistant_marker': None,
        'think_start': None,
        'think_end': None,
    }

    for i, tok in enumerate(tokens):
        if tok == special_ids['bos'] and positions['bos'] is None:
            positions['bos'] = i
        elif tok == special_ids['user_marker'] and positions['user_marker'] is None:
            positions['user_marker'] = i
        elif tok == special_ids['assistant_marker'] and positions['assistant_marker'] is None:
            positions['assistant_marker'] = i
        elif tok == special_ids['think_start'] and positions['think_start'] is None:
            positions['think_start'] = i
        elif tok == special_ids['think_end'] and positions['think_end'] is None:
            positions['think_end'] = i
            break  # Found all markers

    return positions


def classify_position(position: int, markers: dict) -> tuple[str, Optional[float]]:
    """
    Classify which region a token position belongs to.

    Returns: (region_name, thinking_position_or_none)
        - thinking_position is 0.0-1.0 for tokens in thinking region, None otherwise
    """
    # Check single-token special markers first
    if position == markers.get('bos'):
        return 'bos', None
    if position == markers.get('user_marker'):
        return 'user_marker', None
    if position == markers.get('assistant_marker'):
        return 'assistant_marker', None
    if position == markers.get('think_start'):
        return 'think_start', None
    if position == markers.get('think_end'):
        return 'think_end', None

    # Content regions
    assistant_pos = markers.get('assistant_marker')
    think_start_pos = markers.get('think_start')
    think_end_pos = markers.get('think_end')

    # Before assistant marker = question
    if assistant_pos is not None and position < assistant_pos:
        return 'question', None

    # Inside thinking tags
    if think_start_pos is not None and think_end_pos is not None:
        if think_start_pos < position < think_end_pos:
            # Compute relative position within thinking (0 = start, 1 = end)
            thinking_content_start = think_start_pos + 1
            thinking_content_end = think_end_pos - 1
            thinking_length = thinking_content_end - thinking_content_start + 1
            if thinking_length > 0:
                relative_pos = (position - thinking_content_start) / thinking_length
            else:
                relative_pos = 0.5
            return 'thinking', relative_pos

    # After think_end = answer
    if think_end_pos is not None and position > think_end_pos:
        return 'answer', None

    # Fallback (shouldn't happen with well-formed data)
    return 'unknown', None


def precompute_regions(tokens: list[int], markers: dict) -> tuple[list[str], list[Optional[float]]]:
    """Precompute region classification for all positions in a sequence."""
    regions = []
    thinking_positions = []
    for pos in range(len(tokens)):
        region, think_pos = classify_position(pos, markers)
        regions.append(region)
        thinking_positions.append(think_pos)
    return regions, thinking_positions


# =============================================================================
# Feature Collector
# =============================================================================

class FeatureCollector:
    """Collects feature activation statistics across sequences."""

    def __init__(
        self,
        n_layers: int,
        n_features: int,
        top_k: int = 20,
        n_random: int = 10,
        context_before: int = 50,
        context_after: int = 20,
    ):
        self.n_layers = n_layers
        self.n_features = n_features
        self.top_k = top_k
        self.n_random = n_random
        self.context_before = context_before
        self.context_after = context_after

        # Per-feature stats
        self.stats = [
            [FeatureStats(layer_idx=l, feature_idx=f) for f in range(n_features)]
            for l in range(n_layers)
        ]

        # Global counts for normalization
        self.total_tokens = 0
        self.tokens_per_domain: dict[str, int] = defaultdict(int)
        self.tokens_per_region: dict[str, int] = defaultdict(int)
        self.tokens_per_thinking_bin: list[int] = [0] * 10

        # Hook storage
        self._hooks: list = []
        self._layer_activations: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        """Create a forward hook that captures transcoder activations."""
        def hook(module, input, output):
            hidden_states = input[0]  # [batch, seq, d_model]
            with torch.no_grad():
                pre_act = module.transcoder_enc(hidden_states)
                features = torch.relu(pre_act)  # [batch, seq, n_features]
            self._layer_activations[layer_idx] = features[0]  # [seq, n_features]
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
        features_gpu: torch.Tensor,  # full tensor on GPU
        feature_idx: int,
        domain: str,
        region: str,
        thinking_position: Optional[float],
        sequence_idx: int,
    ):
        """Add example to top-k heap and/or random reservoir if appropriate."""
        # Check if this could make it into top-k
        dominated_by_heap = (
            len(stats.top_k_examples) >= self.top_k
            and activation <= stats.top_k_examples[0].activation
        )

        # Reservoir sampling decision for random
        stats.random_seen_count += 1
        add_to_random = False
        random_replace_idx = None
        if len(stats.random_examples) < self.n_random:
            add_to_random = True
        else:
            j = random.randint(0, stats.random_seen_count - 1)
            if j < self.n_random:
                add_to_random = True
                random_replace_idx = j

        # Skip if not going into either buffer
        if dominated_by_heap and not add_to_random:
            return

        # Create example (only fetch context from GPU when actually keeping)
        context_tokens, pos_in_ctx = self._get_context(tokens, position)
        ctx_start = max(0, position - self.context_before)
        ctx_end = min(len(tokens), position + self.context_after + 1)
        # Fetch context window from GPU (only for kept examples)
        context_activations = features_gpu[ctx_start:ctx_end, feature_idx].float().cpu().tolist()

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
        if not dominated_by_heap:
            if len(stats.top_k_examples) < self.top_k:
                heapq.heappush(stats.top_k_examples, example)
            else:
                heapq.heapreplace(stats.top_k_examples, example)

        # Add to random reservoir
        if add_to_random:
            if random_replace_idx is None:
                stats.random_examples.append(example)
            else:
                stats.random_examples[random_replace_idx] = example

    def process_sequence(
        self,
        model,
        tokens: list[int],
        domain: str,
        markers: dict,
        sequence_idx: int,
    ):
        """Process a single sequence and update all feature stats."""
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
            if thinking_positions[pos] is not None:
                bin_idx = min(9, int(thinking_positions[pos] * 10))
                self.tokens_per_thinking_bin[bin_idx] += 1

        # Process each layer
        for layer_idx in range(self.n_layers):
            features_gpu = self._layer_activations[layer_idx]  # [seq_len, n_features] on GPU

            # Find non-zero entries on GPU, transfer only sparse indices/values
            nonzero = torch.nonzero(features_gpu > 0)  # [N, 2] on GPU
            if len(nonzero) == 0:
                continue

            active_positions = nonzero[:, 0].cpu().numpy()  # [N] int64
            active_features = nonzero[:, 1].cpu().numpy()   # [N] int64
            active_values = features_gpu[nonzero[:, 0], nonzero[:, 1]].float().cpu().numpy()  # [N] float32

            # Process all activations
            for i in range(len(active_positions)):
                pos = int(active_positions[i])
                feature_idx = int(active_features[i])
                act = float(active_values[i])

                stats = self.stats[layer_idx][feature_idx]
                region = regions[pos]
                think_pos = thinking_positions[pos]

                # Update counts
                stats.activation_count += 1
                stats.domain_counts[domain] += 1
                stats.region_counts[region] += 1
                if think_pos is not None:
                    bin_idx = min(9, int(think_pos * 10))
                    stats.thinking_position_counts[bin_idx] += 1

                # Maybe add to examples (fetch context from GPU only if needed)
                self._maybe_add_example(
                    stats, act, tokens, pos, features_gpu, feature_idx,
                    domain, region, think_pos, sequence_idx
                )

        # Clear activations
        self._layer_activations = {}


# =============================================================================
# Logit Lens
# =============================================================================

def compute_logit_lens(model, tokenizer, top_k: int = 10) -> list[dict]:
    """
    Compute top/bottom unembed tokens for each feature.

    For each feature, computes decoder @ unembed.T to find which tokens
    the feature most strongly promotes/suppresses.
    """
    print("Computing logit lens...")

    # Get unembedding matrix
    unembed = model.lm_head.weight.data  # [vocab_size, d_model]

    logit_lens_data = []

    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc="Logit lens")):
        mlp = layer.mlp
        # Decoder: [d_model, n_features] - each column is a feature's output direction
        decoder = mlp.transcoder_dec.weight.data  # [d_model, n_features]

        # Compute logits for each feature: unembed @ decoder = [vocab_size, n_features]
        with torch.no_grad():
            logits = unembed @ decoder  # [vocab_size, n_features]

        # Get top and bottom tokens for each feature
        top_vals, top_ids = logits.topk(top_k, dim=0)  # [top_k, n_features]
        bot_vals, bot_ids = logits.topk(top_k, dim=0, largest=False)

        layer_data = {
            'top_ids': top_ids.T.cpu().tolist(),  # [n_features, top_k]
            'top_vals': top_vals.T.cpu().tolist(),
            'bot_ids': bot_ids.T.cpu().tolist(),
            'bot_vals': bot_vals.T.cpu().tolist(),
        }
        logit_lens_data.append(layer_data)

    return logit_lens_data


# =============================================================================
# Export Functions
# =============================================================================

def cantor_pair(x: int, y: int) -> int:
    """Cantor pairing function - maps (layer, feature) to unique integer."""
    return (x + y) * (x + y + 1) // 2 + y


def format_example_for_circuit_tracer(ex: ActivatingExample, tokenizer) -> dict:
    """Format an ActivatingExample for circuit tracer JSON."""
    tokens = [tokenizer.decode([tok_id]) for tok_id in ex.context_tokens]
    return {
        "tokens": tokens,
        "tokens_acts_list": ex.context_activations,
        "train_token_ind": ex.position_in_context,
        "is_repeated_datapoint": False,
    }


def _write_feature_json(args: tuple) -> None:
    """Write a single feature JSON file (for parallel execution)."""
    filepath, feature_json = args
    with open(filepath, 'w') as f:
        json.dump(feature_json, f)


def export_circuit_tracer_json(
    collector: FeatureCollector,
    logit_lens_data: list[dict],
    tokenizer,
    output_dir: Path,
    n_workers: int = 16,
):
    """Export feature data to circuit tracer JSON format."""
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting features to {features_dir}...")

    # First pass: build all JSON objects
    write_tasks = []  # list of (filepath, json_dict)
    skipped = 0

    for layer_idx in tqdm(range(collector.n_layers), desc="Building JSON"):
        layer_logit_lens = logit_lens_data[layer_idx]

        for feature_idx in range(collector.n_features):
            stats = collector.stats[layer_idx][feature_idx]

            # Skip empty features
            if stats.activation_count == 0:
                skipped += 1
                continue

            # Get examples (sorted by activation for top-k)
            top_examples = sorted(stats.top_k_examples, key=lambda x: -x.activation)
            random_examples = stats.random_examples

            # Format examples
            top_formatted = [format_example_for_circuit_tracer(ex, tokenizer) for ex in top_examples]
            random_formatted = [format_example_for_circuit_tracer(ex, tokenizer) for ex in random_examples]

            # Compute activation range from examples
            all_acts = []
            for ex in top_examples + random_examples:
                all_acts.extend(ex.context_activations)
            act_min = min(all_acts) if all_acts else 0.0
            act_max = max(all_acts) if all_acts else 1.0

            # Get logit lens tokens
            top_logits = [tokenizer.decode([tok_id]) for tok_id in layer_logit_lens['top_ids'][feature_idx]]
            bottom_logits = [tokenizer.decode([tok_id]) for tok_id in layer_logit_lens['bot_ids'][feature_idx]]

            # Build JSON
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
                "feature": feature_idx,
            }

            cantor_id = cantor_pair(layer_idx, feature_idx)
            filepath = features_dir / f"{cantor_id}.json"
            write_tasks.append((filepath, feature_json))

    # Second pass: write files in parallel
    print(f"Writing {len(write_tasks)} files with {n_workers} workers...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(tqdm(executor.map(_write_feature_json, write_tasks), total=len(write_tasks), desc="Writing files"))

    print(f"Generated {len(write_tasks)} feature files, skipped {skipped} empty features")


def export_metadata(collector: FeatureCollector, output_dir: Path):
    """Export rich metadata to JSON for analysis."""
    print("Exporting metadata...")

    metadata = {
        # Global counts
        "total_tokens": collector.total_tokens,
        "tokens_per_domain": dict(collector.tokens_per_domain),
        "tokens_per_region": dict(collector.tokens_per_region),
        "tokens_per_thinking_bin": collector.tokens_per_thinking_bin,

        # Per-feature stats
        "features": [],
    }

    for layer_idx in range(collector.n_layers):
        for feature_idx in range(collector.n_features):
            stats = collector.stats[layer_idx][feature_idx]

            if stats.activation_count == 0:
                continue

            total_acts = stats.activation_count

            # Domain distributions
            domain_density = {}
            domain_fraction = {}
            for domain, count in stats.domain_counts.items():
                domain_tokens = collector.tokens_per_domain.get(domain, 0)
                domain_density[domain] = count / domain_tokens if domain_tokens > 0 else 0
                domain_fraction[domain] = count / total_acts

            # Region distributions
            region_density = {}
            region_fraction = {}
            for region, count in stats.region_counts.items():
                region_tokens = collector.tokens_per_region.get(region, 0)
                region_density[region] = count / region_tokens if region_tokens > 0 else 0
                region_fraction[region] = count / total_acts

            # Thinking position distributions
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
                "feature": feature_idx,
                "cantor_id": cantor_pair(layer_idx, feature_idx),
                "activation_count": total_acts,
                "activation_freq": total_acts / collector.total_tokens,
                "domain_density": domain_density,
                "domain_fraction": domain_fraction,
                "region_density": region_density,
                "region_fraction": region_fraction,
                "thinking_position_density": thinking_density,
                "thinking_position_fraction": thinking_fraction,
            }
            metadata["features"].append(feature_meta)

    with open(output_dir / "feature_metadata.json", 'w') as f:
        json.dump(metadata, f)

    print(f"Saved metadata for {len(metadata['features'])} features")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect transcoder feature activations for visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF repo ID or local path to transcoder checkpoint")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Path to validation JSONL (local or hf://)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for feature JSONs and metadata")

    # Optional args
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Tokenizer name/path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (default: all)")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top activating examples per feature")
    parser.add_argument("--n_random", type=int, default=10,
                        help="Number of random samples per feature")
    parser.add_argument("--context_before", type=int, default=50,
                        help="Context tokens before activating token")
    parser.add_argument("--context_after", type=int, default=20,
                        help="Context tokens after activating token")
    parser.add_argument("--max_length", type=int, default=10000,
                        help="Max sequence length (longer sequences truncated)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load model
    print(f"Loading model: {args.model_path}")
    model = Qwen2ForCausalLMWithTranscoder.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    n_layers = len(model.model.layers)
    n_features = model.model.layers[0].mlp.n_features
    print(f"Model: {n_layers} layers, {n_features} features per layer")

    # Load dataset
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

    # Create collector
    collector = FeatureCollector(
        n_layers=n_layers,
        n_features=n_features,
        top_k=args.top_k,
        n_random=args.n_random,
        context_before=args.context_before,
        context_after=args.context_after,
    )

    # Register hooks
    collector.register_hooks(model)

    # Process sequences
    skipped = 0
    for idx in tqdm(range(n_samples), desc="Processing sequences"):
        item = dataset[idx]
        meta = dataset.examples[idx]

        tokens = item['input_ids']
        domain = meta.get('domain', 'unknown')

        # Find special token markers
        markers = find_token_positions(tokens, tokenizer)

        # Validate structure
        if markers['think_start'] is None or markers['think_end'] is None:
            print(f"Warning: Skipping sample {idx} - missing <think> tags")
            skipped += 1
            continue

        collector.process_sequence(model, tokens, domain, markers, idx)

    collector.remove_hooks()

    if skipped > 0:
        print(f"Skipped {skipped} samples due to missing <think> tags")

    # Summary stats
    print(f"\nCollection summary:")
    print(f"  Total tokens: {collector.total_tokens:,}")
    print(f"  Domains: {dict(collector.tokens_per_domain)}")
    print(f"  Regions: {dict(collector.tokens_per_region)}")

    # Compute logit lens
    logit_lens_data = compute_logit_lens(model, tokenizer)

    # Export
    export_circuit_tracer_json(collector, logit_lens_data, tokenizer, output_dir)
    export_metadata(collector, output_dir)

    print(f"\nDone! Output written to {output_dir}")
    print(f"  features/: Circuit tracer JSON files")
    print(f"  feature_metadata.json: Rich metadata for analysis")


if __name__ == "__main__":
    main()
