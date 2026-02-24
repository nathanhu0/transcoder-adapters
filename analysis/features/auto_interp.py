"""
Auto-interpretability with detection-based evaluation.

Takes the expanded JSON format produced by collect_feature_activations or
collect_neuron_activations (feature_metadata.json + features/{cantor_id}.json).

Pipeline per feature:
1. Get examples from "Top activations" and "Random samples" quantiles
2. Use 10 top activating examples for description (with <<<markers>>>)
3. Get 5 random negatives from the validation dataset
4. LLM Query 1: Generate description from 10 examples
5. LLM Query 2a: Detection task with 5 top activating + 5 negatives
6. LLM Query 2b: Detection task with 5 random samples + 5 negatives

Usage:
    python -m analysis.features.auto_interp \
        --input_dir /path/to/circuit_tracing/model_name \
        --data_path /path/to/openthoughts_val.jsonl \
        --output autointerp.json \
        --n_per_layer 100
"""

import argparse
import json
import asyncio
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureResult:
    """Full result for a single feature."""
    layer: int
    feature: int
    cantor_id: int
    # Description
    description: str
    reasoning: str
    # Detection on top activating examples
    detection_top_accuracy: float | None
    detection_top_precision: float | None
    detection_top_recall: float | None
    # Detection on random samples
    detection_random_accuracy: float | None
    detection_random_precision: float | None
    detection_random_recall: float | None
    # Metadata
    n_examples: int
    activation_freq: float


# =============================================================================
# Prompts
# =============================================================================

DESCRIPTION_SYSTEM = """You are a meticulous AI researcher investigating a specific neuron inside a language model. Your task is to describe what causes the neuron to activate.

You will receive text excerpts where the neuron activates. The activating token is marked with <<<token>>>.

Important notes:
- The <<<>>> markers are ONLY for highlighting which token activates - do NOT include <<<>>> in your description
- All examples are from mathematical reasoning contexts, so "math" or "reasoning" alone is NOT a useful description
- Neuron activations can only depend on the marked token and tokens BEFORE it (not after)
- Describe BOTH the general context AND the specific token/word/phrase that activates
- Be extremely specific: look for specific tokens, characters, syntactic positions, semantic patterns, or reasoning steps
- Descriptions should be 10-15 words, no need for complete sentences

Respond in JSON format."""

DESCRIPTION_USER = """Neuron L{layer}F{feature}:

{examples}

Respond with:
{{
    "reasoning": "brief analysis of patterns you see",
    "description": "concise description (10-15 words)"
}}"""

DETECTION_SYSTEM = """You are evaluating whether a neuron description accurately predicts neuron activations.

You will be given:
1. A description of what causes a neuron to activate
2. 10 text snippets (exactly 5 activate the neuron, 5 do not)

For each snippet, predict whether the neuron activates based ONLY on the description.

Respond with a JSON object mapping snippet numbers to predictions:
{"1": true, "2": false, "3": true, ...}"""

DETECTION_USER = """Neuron description: {description}

Snippets (5 activate, 5 don't):
{snippets}

For each snippet, does the neuron activate? Respond with JSON: {{"1": true/false, "2": true/false, ...}}"""


# =============================================================================
# Loading
# =============================================================================

def load_random_snippets(
    data_path: Path,
    tokenizer,
    n_samples: int = 2000,
    max_tokens: int = 71,  # context_before(50) + 1 + context_after(20)
    seed: int = 42,
) -> list[str]:
    """Load random text snippets from validation dataset, truncated to match example length."""
    random.seed(seed)

    all_items = []
    with open(data_path) as f:
        for line in f:
            all_items.append(json.loads(line))

    # Sample subset
    sampled = random.sample(all_items, min(n_samples, len(all_items)))

    snippets = []
    for item in sampled:
        question = item.get("question", item.get("problem", ""))
        response = item.get("response", item.get("solution", ""))
        full_text = f"{question}\n\n{response}"

        # Tokenize and truncate to max_tokens
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
        snippet = tokenizer.decode(token_ids)
        snippets.append(snippet)

    return snippets


def load_metadata(input_dir: Path) -> dict:
    """Load feature_metadata.json."""
    with open(input_dir / "feature_metadata.json") as f:
        return json.load(f)


def load_feature_json(input_dir: Path, cantor_id: int) -> dict | None:
    """Load a single feature JSON file."""
    path = input_dir / "features" / f"{cantor_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def select_features(
    metadata: dict,
    n_per_layer: int,
    min_activation_freq: float = 1e-5,
    min_examples: int = 15,
    seed: int = 42,
) -> list[dict]:
    """Select features uniformly from alive features per layer."""
    random.seed(seed)

    # Group by layer, filter by freq and example count
    by_layer = {}
    for feat in metadata["features"]:
        layer = feat["layer"]
        if feat["activation_freq"] < min_activation_freq:
            continue
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(feat)

    # Sample from each layer
    selected = []
    for layer in sorted(by_layer.keys()):
        layer_feats = by_layer[layer]
        n_sample = min(n_per_layer, len(layer_feats))
        sampled = random.sample(layer_feats, n_sample)
        selected.extend(sampled)
        print(f"  Layer {layer}: {len(layer_feats)} alive, sampled {n_sample}")

    return selected


# =============================================================================
# Formatting
# =============================================================================

def format_example_with_marker(example: dict) -> str:
    """Format example WITH <<<marker>>> for description generation."""
    tokens = example["tokens"]
    highlight_idx = example["train_token_ind"]

    parts = []
    for i, tok in enumerate(tokens):
        if i == highlight_idx:
            parts.append(f"<<<{tok}>>>")
        else:
            parts.append(tok)

    return ''.join(parts)


def format_example_no_marker(example: dict) -> str:
    """Format example WITHOUT marker for detection task."""
    return ''.join(example["tokens"])


def get_random_negatives(
    random_snippets: list[str],
    n_negatives: int,
    rng: random.Random,
) -> list[str]:
    """Get random snippets from the pre-loaded validation dataset."""
    return rng.sample(random_snippets, min(n_negatives, len(random_snippets)))


# =============================================================================
# LLM Processing
# =============================================================================

def compute_detection_metrics(predictions: dict, detect_items: list[tuple[str, bool]]) -> tuple[float, float, float]:
    """Compute accuracy, precision, recall from predictions."""
    tp = fp = tn = fn = 0
    for i, (_, is_positive) in enumerate(detect_items, 1):
        pred = predictions.get(str(i), False)
        if isinstance(pred, str):
            pred = pred.lower() == "true"

        if is_positive and pred:
            tp += 1
        elif is_positive and not pred:
            fn += 1
        elif not is_positive and pred:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn) / len(detect_items)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return accuracy, precision, recall


async def run_detection_task(
    client: AsyncOpenAI,
    description: str,
    positives: list[dict],
    negatives: list[str],
    rng: random.Random,
    model: str,
) -> tuple[float | None, float | None, float | None]:
    """Run a single detection task, return (accuracy, precision, recall)."""
    detect_items = []
    for ex in positives:
        detect_items.append((format_example_no_marker(ex), True))
    for neg_text in negatives:
        detect_items.append((neg_text, False))

    rng.shuffle(detect_items)

    snippets_text = "\n\n".join(
        f"Snippet {i+1}:\n{text}"
        for i, (text, _) in enumerate(detect_items)
    )

    detection_user = DETECTION_USER.format(
        description=description,
        snippets=snippets_text,
    )

    try:
        detect_response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DETECTION_SYSTEM},
                {"role": "user", "content": detection_user},
            ],
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        content = detect_response.choices[0].message.content
        assert content is not None
        predictions = json.loads(content)
        return compute_detection_metrics(predictions, detect_items)
    except Exception as e:
        return None, None, None


async def process_feature(
    client: AsyncOpenAI,
    feat_meta: dict,
    feature_json: dict,
    random_snippets: list[str],
    rng: random.Random,
    model: str = "gpt-4o-mini",
    n_description: int = 10,
    n_detection: int = 5,
) -> FeatureResult | None:
    """Process a single feature: generate description + two detection evaluations."""

    # Get examples by quantile
    top_examples = []
    random_examples = []
    for q in feature_json.get("examples_quantiles", []):
        if q.get("quantile_name") == "Top activations":
            top_examples = q.get("examples", [])
        elif q.get("quantile_name") == "Random samples":
            random_examples = q.get("examples", [])

    # Need enough top examples for description + detection_top
    if len(top_examples) < n_description:
        return None

    # Split examples
    desc_examples = top_examples[:n_description]
    detect_top_positives = top_examples[:n_detection]  # Can overlap with description
    detect_random_positives = random_examples[:n_detection] if len(random_examples) >= n_detection else None

    # === LLM Query 1: Description ===
    examples_text = "\n\n".join(
        f"Excerpt {i+1} (act={ex['tokens_acts_list'][ex['train_token_ind']]:.2f}):\n{format_example_with_marker(ex)}"
        for i, ex in enumerate(desc_examples)
    )

    description_user = DESCRIPTION_USER.format(
        layer=feat_meta["layer"],
        feature=feat_meta["feature"],
        examples=examples_text,
    )

    try:
        desc_response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DESCRIPTION_SYSTEM},
                {"role": "user", "content": description_user},
            ],
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        content = desc_response.choices[0].message.content
        assert content is not None
        desc_result = json.loads(content)
        description = desc_result.get("description", "")
        reasoning = desc_result.get("reasoning", "")
    except Exception as e:
        print(f"Description error L{feat_meta['layer']}F{feat_meta['feature']}: {e}")
        return None

    # === LLM Query 2a: Detection on top activating examples ===
    negatives_top = get_random_negatives(random_snippets, n_detection, rng)
    top_acc, top_prec, top_rec = await run_detection_task(
        client, description, detect_top_positives, negatives_top, rng, model
    )

    # === LLM Query 2b: Detection on random samples (skip if no random samples) ===
    if detect_random_positives is not None:
        negatives_random = get_random_negatives(random_snippets, n_detection, rng)
        rand_acc, rand_prec, rand_rec = await run_detection_task(
            client, description, detect_random_positives, negatives_random, rng, model
        )
    else:
        rand_acc, rand_prec, rand_rec = None, None, None

    return FeatureResult(
        layer=feat_meta["layer"],
        feature=feat_meta["feature"],
        cantor_id=feat_meta["cantor_id"],
        description=description,
        reasoning=reasoning,
        detection_top_accuracy=top_acc,
        detection_top_precision=top_prec,
        detection_top_recall=top_rec,
        detection_random_accuracy=rand_acc,
        detection_random_precision=rand_prec,
        detection_random_recall=rand_rec,
        n_examples=len(top_examples) + len(random_examples),
        activation_freq=feat_meta["activation_freq"],
    )


async def process_features(
    client: AsyncOpenAI,
    input_dir: Path,
    features: list[dict],
    random_snippets: list[str],
    model: str = "gpt-4o-mini",
    max_concurrent: int = 50,
    seed: int = 42,
) -> list[FeatureResult]:
    """Process features with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(feat_meta: dict, idx: int):
        async with semaphore:
            feature_json = load_feature_json(input_dir, feat_meta["cantor_id"])
            if feature_json is None:
                return None
            rng = random.Random(seed + idx)
            return await process_feature(
                client, feat_meta, feature_json, random_snippets, rng, model
            )

    tasks = [process_one(f, i) for i, f in enumerate(features)]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing features")
    return [r for r in results if r is not None]


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Auto-interpretability with detection-based evaluation"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Circuit tracer output directory")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to validation JSONL for random negatives")
    parser.add_argument("--output", type=str, default="autointerp.json",
                       help="Output JSON path (relative to input_dir or absolute)")
    parser.add_argument("--n_per_layer", type=int, default=100,
                       help="Number of features to sample per layer")
    # 6e-7 ensures ~10 examples in 16M token val set (need 10 top + 5 random)
    parser.add_argument("--min_freq", type=float, default=6e-7,
                       help="Minimum activation frequency")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="OpenAI model")
    parser.add_argument("--max_concurrent", type=int, default=100,
                       help="Max concurrent API calls")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B",
                       help="Tokenizer for truncating snippets")
    parser.add_argument("--max_tokens", type=int, default=71,
                       help="Max tokens for snippets (context_before + 1 + context_after)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    data_path = Path(args.data_path)

    print(f"Loading tokenizer {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading random snippets from {data_path} (max {args.max_tokens} tokens)...")
    random_snippets = load_random_snippets(
        data_path, tokenizer, max_tokens=args.max_tokens, seed=args.seed
    )
    print(f"  Loaded {len(random_snippets)} snippets for negatives")

    print(f"Loading metadata from {input_dir}...")
    metadata = load_metadata(input_dir)
    all_features = metadata["features"]
    print(f"  Total features: {len(all_features)}")

    print(f"\nSelecting {args.n_per_layer} features per layer (min_freq={args.min_freq:.0e})...")
    selected = select_features(
        metadata,
        n_per_layer=args.n_per_layer,
        min_activation_freq=args.min_freq,
        seed=args.seed,
    )
    print(f"Selected {len(selected)} total features")

    print(f"\nProcessing with {args.model}...")
    client = AsyncOpenAI()
    results = await process_features(
        client, input_dir, selected, random_snippets,
        model=args.model,
        max_concurrent=args.max_concurrent,
        seed=args.seed,
    )
    print(f"Processed {len(results)} features")

    if not results:
        print("No results!")
        return

    # Stats
    print("\n=== Detection on Top Activations ===")
    top_accs = [r.detection_top_accuracy for r in results if r.detection_top_accuracy is not None]
    if top_accs:
        print(f"Accuracy:  mean={sum(top_accs)/len(top_accs):.2f}, min={min(top_accs):.2f}, max={max(top_accs):.2f}")
    top_precs = [r.detection_top_precision for r in results if r.detection_top_precision is not None]
    if top_precs:
        print(f"Precision: mean={sum(top_precs)/len(top_precs):.2f}")
    top_recs = [r.detection_top_recall for r in results if r.detection_top_recall is not None]
    if top_recs:
        print(f"Recall:    mean={sum(top_recs)/len(top_recs):.2f}")

    print("\n=== Detection on Random Samples ===")
    rand_accs = [r.detection_random_accuracy for r in results if r.detection_random_accuracy is not None]
    if rand_accs:
        print(f"Accuracy:  mean={sum(rand_accs)/len(rand_accs):.2f}, min={min(rand_accs):.2f}, max={max(rand_accs):.2f}")
    rand_precs = [r.detection_random_precision for r in results if r.detection_random_precision is not None]
    if rand_precs:
        print(f"Precision: mean={sum(rand_precs)/len(rand_precs):.2f}")
    rand_recs = [r.detection_random_recall for r in results if r.detection_random_recall is not None]
    if rand_recs:
        print(f"Recall:    mean={sum(rand_recs)/len(rand_recs):.2f}")

    # Save
    output = {
        "metadata": {
            "input_dir": str(input_dir),
            "data_path": str(data_path),
            "model": args.model,
            "tokenizer": args.tokenizer,
            "max_tokens": args.max_tokens,
            "n_per_layer": args.n_per_layer,
            "min_freq": args.min_freq,
            "seed": args.seed,
            "n_processed": len(results),
            "mean_detection_top_accuracy": sum(top_accs) / len(top_accs) if top_accs else None,
            "mean_detection_random_accuracy": sum(rand_accs) / len(rand_accs) if rand_accs else None,
        },
        "features": [asdict(r) for r in results],
    }

    output_path = input_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print samples
    print("\n" + "=" * 60)
    print("Sample results:")
    print("=" * 60)
    for r in results[:5]:
        top_acc = f"{r.detection_top_accuracy:.2f}" if r.detection_top_accuracy else "N/A"
        rand_acc = f"{r.detection_random_accuracy:.2f}" if r.detection_random_accuracy else "N/A"
        print(f"\nL{r.layer}F{r.feature} (top_acc={top_acc}, rand_acc={rand_acc}):")
        print(f"  {r.description}")


if __name__ == "__main__":
    asyncio.run(main())
