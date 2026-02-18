#!/usr/bin/env python3
"""
LLM-judge feature classification.

Takes the expanded JSON format produced by collect_feature_activations or
collect_neuron_activations (feature_metadata.json + features/{cantor_id}.json)
and classifies each feature into interpretable categories using an LLM.

Two-level classification:

Level 1 - Feature Domain:
  - language: General text/language modeling (punctuation, conjunctions)
  - domain: Math/code domain knowledge (formulas, technical terms)
  - reasoning: Reasoning-specific behaviors (uncertainty, planning, checking)
  - uninterpretable: No clear pattern

Level 2 - Mechanism (for reasoning features only):
  - narrow_input: Fires on specific tokens, broad output effect
  - narrow_output: Fires broadly, promotes specific tokens
  - input_output: Both narrow input AND narrow output
  - abstract: High-level reasoning concept, not simple I/O

Usage:
    python -m analysis.features.classify_features \
        --input_dir /path/to/circuit_tracing/model_name \
        --output feature_classifications.json \
        --n_per_layer 50

    # Load and analyze cached results:
    python -m analysis.features.classify_features \
        --input_dir /path/to/circuit_tracing/model_name \
        --load feature_classifications.json
"""

#%%
import argparse
import json
import asyncio
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Literal

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

#%%
# =============================================================================
# Configuration
# =============================================================================

# Level 1: Feature domain categories
DOMAIN_CATEGORIES = [
    "language",        # General text/language modeling (punctuation, conjunctions, articles)
    "domain",          # Math/code/reasoning domain knowledge (formulas, technical terms)
    "reasoning",       # Reasoning-specific behaviors (uncertainty, planning, checking)
    "uninterpretable", # No clear pattern, noisy, or incomprehensible
]

# Level 2a: Mechanism (for reasoning features only)
REASONING_MECHANISMS = [
    "output",          # Clear output pattern (promotes specific tokens), input may vary
    "input_simple",    # Token-level input: fires on specific tokens regardless of context
    "input_abstract",  # Context-level input: firing depends on broader context, not just the token
]

# Level 2b: Domain type (for domain features only)
DOMAIN_TYPES = [
    "math",            # Mathematical notation, terms, equations
    "science",         # Scientific formulas, chemistry, physics terminology
    "code",            # Programming syntax, keywords, operators
]

#%%
# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureClassification:
    """Classification result for a single feature."""
    layer: int
    feature: int
    cantor_id: int

    # Level 1: Feature domain
    category: Literal["language", "domain", "reasoning", "uninterpretable"]
    confidence: Literal["high", "medium", "low"]

    # Level 2a: Mechanism (for reasoning features only)
    mechanism: Optional[Literal["output", "input_simple", "input_abstract"]]

    # Level 2b: Domain type (for domain features only)
    domain_type: Optional[Literal["math", "science", "code"]]

    # Descriptions
    category_description: str  # Brief explanation of why this category
    mechanism_description: Optional[str]  # For reasoning: explanation of the mechanism

    # What triggers it (for input_simple or input_abstract)
    input_pattern: Optional[str]  # e.g., "fires on 'wait'" or "fires at contradiction points"

    # What it promotes (especially for output mechanism)
    output_pattern: Optional[str]  # e.g., "promotes 'Wait'", "promotes hesitation words"

    # Raw LLM output
    llm_reasoning: str

    # Metadata
    n_examples: int
    activation_freq: Optional[float]
    top_logits: list[str]


#%%
# =============================================================================
# Prompts
# =============================================================================

CLASSIFICATION_SYSTEM = """You are a meticulous AI researcher analyzing neurons (features) in a sparse autoencoder trained on a mathematical reasoning model (DeepSeek-R1-Distill).

Your task is to classify features based on their activation patterns (where they fire) and output behavior (what tokens they promote).

## Level 1: Feature Domain

### "language" - General Language/Text Modeling
Features related to general language patterns that make text flow, not specific to math/code or reasoning model behaviors.

Key distinction from "reasoning": If a feature fires on generic language but ONLY in reasoning-model-specific contexts (e.g., triggers uncertainty, self-correction), classify it as "reasoning" not "language".

Examples of language features:
- Punctuation (commas, periods, quotes)
- Conjunctions (and, but, or, because)
- Articles and determiners (a, an, the)
- Pronouns and basic syntax
- Generic formatting (indentation, spacing)
- Connector/flow words: "therefore", "hence", "thus", "so", "then", "next"
- Standard prose transitions that make language flow smoothly

### "domain" - Math/Science/Code Technical Knowledge
Features encoding domain-specific technical vocabulary, notation, or patterns. This is about CONTENT knowledge, not language flow.

If you classify as "domain", also specify the domain_type:
- "math": Mathematical notation, equations, variables, math terms (∑, ∫, =, "derivative", "polynomial", "let x be")
- "science": Scientific formulas, chemistry, physics terminology (chemical formulas, physical constants, scientific notation)
- "code": Programming syntax, keywords, operators ("def", "return", "if", brackets, indentation patterns)

Examples:
- Mathematical notation (∑, ∫, ∀, ∃, =, +, numbers, variables) → math
- Code syntax (keywords like "def", "return", "if", operators, brackets) → code
- Technical math terms ("recursion", "derivative", "polynomial", "function", "matrix") → math
- Scientific formulas and notation → science
- Domain-specific structural patterns (equation layout, code blocks, LaTeX) → math or code
- Math setup phrases: "let x be", "suppose", "given that", "define" → math

### "reasoning" - Reasoning Model Behaviors
Features related to the unique behaviors of REASONING MODELS (like DeepSeek-R1), NOT general problem-solving.

KEY DISTINCTION:
- "language": Connector words and flow language ("therefore", "hence", "so", "then") - these make text flow but aren't unique to reasoning models.
- "domain": Technical math/code vocabulary and notation ("function", "derivative", "let x be") - this is content knowledge.
- "reasoning": Behaviors UNIQUE to reasoning models - the verbose, self-reflective, uncertainty-expressing style from RL/distillation training.

IMPORTANT: All examples are collected from reasoning traces, so co-occurrence is NOT enough. The feature must capture something unique to the REASONING MODEL STYLE:
1. Explicit uncertainty/hedging ("Hmm", "Wait", "I think", "maybe", "actually")
2. Self-correction and backtracking ("No, that's wrong", "Let me reconsider", "I made an error")
3. Metacognitive commentary ("Let me think about this", "I need to be careful here")
4. Verification behaviors ("Let me check", "Does this make sense?", "Sanity check")
5. The characteristic "think out loud" verbosity of reasoning models

NOT reasoning:
- Connector words ("therefore", "hence", "so", "thus") → classify as "language"
- Technical terms ("the function", "derivative", "let x be") → classify as "domain"

Examples of reasoning features:
- Uncertainty: "hmm", "wait", "actually", "I'm not sure"
- Self-correction: "no wait", "that's wrong", "let me reconsider"
- Metacognition: "I need to think about", "this is tricky"
- Verification: "let me verify", "checking my work", "does this make sense"
- Features that PROMOTE these reasoning-model-specific tokens

### "uninterpretable" - No Clear Pattern
The feature's firing pattern or role is unclear, even if the examples share surface-level similarities.

Signs to classify as uninterpretable:
- Examples share a theme (e.g., all math text) but you can't identify WHAT specifically triggers activation
- The highlighted token varies without a clear unifying pattern
- Top logits don't relate coherently to the activation pattern

As a very rough guide, prior work on SAE interpretability finds that typically 10-30% of features are uninterpretable. Do not anchor to this number, but don't force patterns that aren't there.

## Level 2: Mechanism (ONLY for "reasoning" features)

If you classified the feature as "reasoning", also classify its mechanism:

### "output" - Output Feature
The defining characteristic is WHAT it promotes. Has a clear output pattern (promotes specific tokens like "Wait", "Hmm", hesitation words). The input may vary but often has some pattern too.

How to identify:
- TOP LOGITS show it consistently promotes specific reasoning-related tokens
- The main story is "this feature promotes X" (input context is secondary)
- Most reasoning features that promote uncertainty/hesitation tokens fall here

Example: Promotes "Wait", "Hmm", "Hold on" - fires at various transition points but the key behavior is promoting these tokens.

### "input_simple" - Simple Input Feature
The defining characteristic is a simple TOKEN-LEVEL input pattern. Fires on specific tokens regardless of surrounding context.

How to identify:
- Fires on the SAME or SIMILAR tokens across examples (e.g., "wait", "Wait", "waiting")
- The token alone determines whether it fires - context doesn't matter
- Output may or may not be clear

Example: Fires on "actually" in any context within reasoning text.

### "input_abstract" - Abstract Input Feature
The defining characteristic is a CONTEXT-DEPENDENT input pattern. The same token might fire in some contexts but not others - broader context determines firing.

How to identify:
- Varied tokens across examples, but a consistent CONTEXTUAL theme
- The token alone does NOT determine firing - context matters
- Examples: "doesn't" only at logical contradictions, various planning phrases, transition points

Example: Fires on "doesn't" but ONLY when arriving at a logical contradiction (not every "doesn't").
Example: Fires on various phrases related to planning/strategizing (unified by concept, not token).

## Output Format

```json
{
    "reasoning": "Your step-by-step analysis...",
    "category": "language" | "domain" | "reasoning" | "uninterpretable",
    "confidence": "high" | "medium" | "low",
    "category_description": "Brief explanation of why this category",

    // ONLY include if category is "domain":
    "domain_type": "math" | "science" | "code",

    // ONLY include if category is "reasoning":
    "mechanism": "output" | "input_simple" | "input_abstract",
    "mechanism_description": "Explanation of the mechanism",
    "input_pattern": "What triggers it (for input_simple or input_abstract)",
    "output_pattern": "What it promotes (especially for output mechanism)"
}
```"""

CLASSIFICATION_USER = """Analyze Feature L{layer}F{feature}:

## Top Logits (tokens this feature promotes when active):
{top_logits}

## Activating Examples (<<<token>>> marks where feature activates):
{examples}

First, classify this feature's domain (language, domain, reasoning, or uninterpretable).
If it's a reasoning feature, also classify its mechanism (output, input_simple, or input_abstract)."""


#%%
# =============================================================================
# Loading
# =============================================================================

def load_metadata(input_dir: Path) -> dict:
    """Load feature_metadata.json."""
    with open(input_dir / "feature_metadata.json") as f:
        return json.load(f)


def load_feature_json(input_dir: Path, cantor_id: int) -> Optional[dict]:
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
    seed: int = 42,
) -> list[dict]:
    """Select features uniformly from alive features per layer."""
    random.seed(seed)

    by_layer = {}
    for feat in metadata["features"]:
        layer = feat["layer"]
        if feat.get("activation_freq", 1.0) < min_activation_freq:
            continue
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(feat)

    selected = []
    for layer in sorted(by_layer.keys()):
        layer_feats = by_layer[layer]
        n_sample = min(n_per_layer, len(layer_feats))
        sampled = random.sample(layer_feats, n_sample)
        selected.extend(sampled)
        print(f"  Layer {layer}: {len(layer_feats)} alive, sampled {n_sample}")

    return selected


#%%
# =============================================================================
# Formatting
# =============================================================================

def format_example_with_marker(example: dict) -> str:
    """Format example WITH <<<marker>>> for classification."""
    tokens = example["tokens"]
    highlight_idx = example["train_token_ind"]

    parts = []
    for i, tok in enumerate(tokens):
        if i == highlight_idx:
            parts.append(f"<<<{tok}>>>")
        else:
            parts.append(tok)

    return ''.join(parts)


def get_top_examples(feature_data: dict, n: int = 10) -> list[dict]:
    """Get top activation examples."""
    for quantile in feature_data.get('examples_quantiles', []):
        if quantile.get('quantile_name') == 'Top activations':
            return quantile.get('examples', [])[:n]
    return []


#%%
# =============================================================================
# LLM Processing
# =============================================================================

async def classify_feature(
    client: AsyncOpenAI,
    feat_meta: dict,
    feature_json: dict,
    model: str = "gpt-4o-mini",
    n_examples: int = 10,
) -> Optional[FeatureClassification]:
    """Classify a single feature using LLM."""

    examples = get_top_examples(feature_json, n_examples)
    if len(examples) < 5:
        return None

    # Format examples
    examples_text = "\n\n".join(
        f"Example {i+1}:\n{format_example_with_marker(ex)}"
        for i, ex in enumerate(examples)
    )

    # Format top logits
    top_logits = feature_json.get('top_logits', [])[:15]
    top_logits_text = ", ".join(f'"{t}"' for t in top_logits)

    user_prompt = CLASSIFICATION_USER.format(
        layer=feat_meta["layer"],
        feature=feat_meta["feature"],
        top_logits=top_logits_text,
        examples=examples_text,
    )

    # Retry with exponential backoff
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            # Combine system and user prompt for responses API
            full_prompt = f"{CLASSIFICATION_SYSTEM}\n\n---\n\n{user_prompt}\n\nRespond with a JSON object."

            response = await client.responses.create(
                model=model,
                input=full_prompt,
            )

            # Parse JSON from response text (may be wrapped in ```json blocks)
            response_text = response.output_text.strip()
            if response_text.startswith("```"):
                # Remove markdown code blocks
                lines = response_text.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                response_text = "\n".join(lines)
            result = json.loads(response_text)
            break  # Success

        except json.JSONDecodeError as e:
            # JSON parsing error - retry might help if model gives different response
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"JSON error L{feat_meta['layer']}F{feat_meta['feature']}, retry {attempt+1} in {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                print(f"JSON error L{feat_meta['layer']}F{feat_meta['feature']} (final): {e}")
                return None

        except Exception as e:
            # API error (rate limit, network, etc)
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str or "limit" in error_str or "429" in error_str

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) * (3 if is_rate_limit else 1)
                print(f"API error L{feat_meta['layer']}F{feat_meta['feature']}, retry {attempt+1} in {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                print(f"API error L{feat_meta['layer']}F{feat_meta['feature']} (final): {e}")
                return None
    else:
        return None  # All retries failed

    category = result.get("category", "uninterpretable")
    is_reasoning = category == "reasoning"
    is_domain = category == "domain"

    return FeatureClassification(
        layer=feat_meta["layer"],
        feature=feat_meta["feature"],
        cantor_id=feat_meta["cantor_id"],
        category=category,
        confidence=result.get("confidence", "low"),
        mechanism=result.get("mechanism") if is_reasoning else None,
        domain_type=result.get("domain_type") if is_domain else None,
        category_description=result.get("category_description", ""),
        mechanism_description=result.get("mechanism_description") if is_reasoning else None,
        input_pattern=result.get("input_pattern") if is_reasoning else None,
        output_pattern=result.get("output_pattern") if is_reasoning else None,
        llm_reasoning=result.get("reasoning", ""),
        n_examples=len(examples),
        activation_freq=feat_meta.get("activation_freq"),
        top_logits=top_logits,
    )


async def process_features(
    client: AsyncOpenAI,
    input_dir: Path,
    features: list[dict],
    model: str = "gpt-4o-mini",
    max_concurrent: int = 50,
) -> list[FeatureClassification]:
    """Process features with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(feat_meta: dict):
        async with semaphore:
            feature_json = load_feature_json(input_dir, feat_meta["cantor_id"])
            if feature_json is None:
                return None
            return await classify_feature(client, feat_meta, feature_json, model)

    tasks = [process_one(f) for f in features]
    results = await tqdm_asyncio.gather(*tasks, desc="Classifying features")
    return [r for r in results if r is not None]


#%%
# =============================================================================
# Analysis
# =============================================================================

def summarize_results(results: list[FeatureClassification]):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total classified: {len(results)}")

    # Level 1: Category counts
    language = [r for r in results if r.category == "language"]
    domain = [r for r in results if r.category == "domain"]
    reasoning = [r for r in results if r.category == "reasoning"]
    uninterpretable = [r for r in results if r.category == "uninterpretable"]

    print(f"\n=== Level 1: Feature Domain ===")
    print(f"  Language:        {len(language)} ({100*len(language)/len(results):.1f}%)")
    print(f"  Domain:          {len(domain)} ({100*len(domain)/len(results):.1f}%)")
    print(f"  Reasoning:       {len(reasoning)} ({100*len(reasoning)/len(results):.1f}%)")
    print(f"  Uninterpretable: {len(uninterpretable)} ({100*len(uninterpretable)/len(results):.1f}%)")

    # Confidence
    high_conf = [r for r in results if r.confidence == "high"]
    med_conf = [r for r in results if r.confidence == "medium"]
    low_conf = [r for r in results if r.confidence == "low"]

    print(f"\n=== Confidence ===")
    print(f"  High:   {len(high_conf)} ({100*len(high_conf)/len(results):.1f}%)")
    print(f"  Medium: {len(med_conf)} ({100*len(med_conf)/len(results):.1f}%)")
    print(f"  Low:    {len(low_conf)} ({100*len(low_conf)/len(results):.1f}%)")

    # Level 2a: Domain type breakdown (for domain features)
    if domain:
        print(f"\n=== Level 2: Domain Types ===")
        type_counts = {}
        for r in domain:
            t = r.domain_type or "unspecified"
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c} ({100*c/len(domain):.1f}%)")

    # Level 2b: Mechanism breakdown (for reasoning features)
    if reasoning:
        print(f"\n=== Level 2: Reasoning Mechanisms ===")
        mech_counts = {}
        for r in reasoning:
            m = r.mechanism or "unspecified"
            mech_counts[m] = mech_counts.get(m, 0) + 1
        for m, c in sorted(mech_counts.items(), key=lambda x: -x[1]):
            print(f"  {m}: {c} ({100*c/len(reasoning):.1f}%)")

    # Sample features by category
    if language:
        print(f"\n=== Sample Language Features ===")
        for r in language[:3]:
            print(f"  L{r.layer}F{r.feature}: {r.category_description}")

    if domain:
        print(f"\n=== Sample Domain Features ===")
        for r in domain[:3]:
            dtype = f"[{r.domain_type}]" if r.domain_type else ""
            print(f"  L{r.layer}F{r.feature} {dtype}: {r.category_description}")

    if reasoning:
        print(f"\n=== Sample Reasoning Features ===")
        for r in reasoning[:5]:
            mech_str = f"[{r.mechanism}]" if r.mechanism else ""
            desc = r.mechanism_description or r.category_description
            print(f"  L{r.layer}F{r.feature} {mech_str}: {desc}")


def load_results(path: Path) -> list[FeatureClassification]:
    """Load results from JSON."""
    with open(path) as f:
        data = json.load(f)

    results = []
    for r in data.get("features", []):
        results.append(FeatureClassification(**r))

    return results


#%%
# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Classify features using LLM judge")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory with feature_metadata.json and features/*.json")
    parser.add_argument("--output", type=str, default="feature_classifications.json",
                       help="Output JSON path (relative to input_dir or absolute)")
    parser.add_argument("--n_per_layer", type=int, default=50,
                       help="Number of features to sample per layer")
    parser.add_argument("--min_freq", type=float, default=1e-7,
                       help="Minimum activation frequency")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                       help="OpenAI model")
    parser.add_argument("--max_concurrent", type=int, default=200,
                       help="Max concurrent API calls")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--load", type=str, default=None,
                       help="Load and analyze existing results instead of computing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Load mode
    if args.load:
        load_path = input_dir / args.load if not Path(args.load).is_absolute() else Path(args.load)
        print(f"Loading results from {load_path}...")
        results = load_results(load_path)
        summarize_results(results)
        return

    # Compute mode
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

    print(f"\nClassifying with {args.model}...")
    client = AsyncOpenAI()
    results = await process_features(
        client, input_dir, selected,
        model=args.model,
        max_concurrent=args.max_concurrent,
    )
    print(f"Classified {len(results)} features")

    if not results:
        print("No results!")
        return

    summarize_results(results)

    # Save
    output = {
        "metadata": {
            "input_dir": str(input_dir),
            "model": args.model,
            "n_per_layer": args.n_per_layer,
            "min_freq": args.min_freq,
            "seed": args.seed,
            "n_classified": len(results),
        },
        "features": [asdict(r) for r in results],
    }

    output_path = input_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
