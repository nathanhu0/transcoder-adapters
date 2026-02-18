#!/usr/bin/env python3
"""Filter OpenThoughts with stratified sampling by domain.

Streams from open-thoughts/OpenThoughts3-1.2M on HuggingFace and produces
train/val JSONL splits with stratified domain sampling.

Strategy:
1. Stratified sampling with fixed ratios (70% math, 20% code, 10% science)
2. Prefer complete responses under soft_max_tokens
3. Supplement with longer examples (to be truncated) if needed

Command used to generate the dataset for this project:

    python misc_scripts/filter_openthoughts_stratified.py \
        --n_train 50000 --n_val 5000 --soft_max_tokens 10000

This produces:
    stratified_n55000_t10000_s42_train.jsonl  (49,952 examples, ~1.3GB)
    stratified_n55000_t10000_s42_val.jsonl    (4,996 examples, ~131MB)

Source dataset: open-thoughts/OpenThoughts3-1.2M
"""

import json
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Fixed domain ratios (matching original OpenThoughts paper)
# Math: 850K, Code: 250K, Science: 100K out of 1.2M
DOMAIN_RATIOS = {
    'math': 0.708,
    'code': 0.208,
    'science': 0.083,
}


def check_completeness(text, domain=None):
    """Check if response is complete.

    Just requires <think>...</think> in order.
    (boxed{} not required - loses too many science/code examples)
    """
    # Find <think>
    start_pos = text.find("<think>")
    if start_pos == -1:
        return False

    # Find </think> after <think>
    end_pos = text.find("</think>", start_pos + len("<think>"))
    return end_pos != -1


# DeepSeek R1 Distill format tokens (matching dataset.py)
DEEPSEEK_USER_TOKEN = "<｜User｜>"
DEEPSEEK_ASSISTANT_TOKEN = "<｜Assistant｜>"


def format_for_tokenization(conv, tokenizer, format="deepseek"):
    """Format conversation for tokenization."""
    prompt = conv[0]['value']
    response = conv[1]['value']

    if format == "deepseek":
        return f"{DEEPSEEK_USER_TOKEN}{prompt}{DEEPSEEK_ASSISTANT_TOKEN}{response}"
    else:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)


def batch_tokenize_lengths(examples, tokenizer, format="deepseek"):
    """Batch tokenize examples and return lengths. Much faster than one-by-one."""
    texts = [format_for_tokenization(ex['conversations'], tokenizer, format) for ex in examples]
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    return [len(ids) for ids in encoded['input_ids']]


def stream_and_collect_by_domain(
    target_per_domain,
    soft_max_tokens,
    tokenizer,
    format="deepseek",
    tokenize_batch_size=256,
    buffer_size=100000,
    seed=42,
):
    """Stream dataset and collect examples per domain with length preference.

    For each domain, collects:
    - Priority: complete + under soft_max_tokens
    - Fallback: complete + over soft_max_tokens (reservoir sampled for truncation)

    Uses batch tokenization for speed.
    """
    import random
    rng = random.Random(seed + 1)  # different seed than shuffle
    long_counts = {d: 0 for d in DOMAIN_RATIOS.keys()}  # track total seen for reservoir

    print(f"Target per domain: {target_per_domain}")
    print(f"Soft max tokens: {soft_max_tokens}")
    print(f"Tokenization format: {format}")

    # Load streaming dataset
    print("Loading dataset (streaming)...")
    dataset = load_dataset("open-thoughts/OpenThoughts3-1.2M", streaming=True)
    print(f"Dataset has {dataset['train'].n_shards} shards")
    shuffled = dataset['train'].shuffle(seed=seed, buffer_size=buffer_size)

    # Collectors per domain: {domain: {'short': [], 'long': []}}
    collected = {d: {'short': [], 'long': []} for d in DOMAIN_RATIOS.keys()}

    total_processed = 0
    dataset_iter = iter(shuffled)

    def domain_complete(domain):
        """Check if we have enough SHORT examples for this domain."""
        target = target_per_domain[domain]
        return len(collected[domain]['short']) >= target

    def all_complete():
        return all(domain_complete(d) for d in DOMAIN_RATIOS.keys())

    print("\nCollecting examples...")
    pbar = tqdm(desc="Processing")

    while not all_complete():
        # Collect a batch of candidates (pass domain + completeness checks)
        batch = []
        while len(batch) < tokenize_batch_size:
            try:
                example = next(dataset_iter)
            except StopIteration:
                print("Reached end of dataset")
                break

            total_processed += 1
            pbar.update(1)

            domain = example['domain']
            if domain not in DOMAIN_RATIOS:
                continue
            if domain_complete(domain):
                continue

            # Check completeness (domain-specific)
            assistant_response = example['conversations'][1]['value']
            if not check_completeness(assistant_response, domain):
                continue

            batch.append(example)

        if not batch:
            break

        # Batch tokenize for speed
        token_lengths = batch_tokenize_lengths(batch, tokenizer, format)

        # Route each example based on length
        for example, token_length in zip(batch, token_lengths):
            domain = example['domain']
            target = target_per_domain[domain]
            n_short = len(collected[domain]['short'])

            if token_length <= soft_max_tokens:
                # Always take short examples up to target
                if n_short < target:
                    collected[domain]['short'].append(example)
            else:
                # Collect long examples (will trim periodically)
                n_long = len(collected[domain]['long'])
                max_reservoir = max(0, target - n_short)
                long_counts[domain] += 1

                if max_reservoir == 0:
                    continue

                if n_long < max_reservoir:
                    collected[domain]['long'].append(example)
                else:
                    # Reservoir sampling: replace with probability max_reservoir / total_seen
                    j = rng.randint(0, long_counts[domain] - 1)
                    if j < max_reservoir:
                        collected[domain]['long'][j] = example

        # Trim reservoirs every 10k examples
        if total_processed % 10000 == 0:
            for d in DOMAIN_RATIOS.keys():
                n_short_d = len(collected[d]['short'])
                max_needed = max(0, target_per_domain[d] - n_short_d)
                if len(collected[d]['long']) > max_needed:
                    rng.shuffle(collected[d]['long'])
                    collected[d]['long'] = collected[d]['long'][:max_needed]

        # Status update every 50k
        if total_processed % 50000 == 0:
            status = []
            for d in DOMAIN_RATIOS.keys():
                ns, nl = len(collected[d]['short']), len(collected[d]['long'])
                t = target_per_domain[d]
                status.append(f"{d}: {ns}+{nl}/{t}")
            pbar.set_postfix_str(" | ".join(status))

    pbar.close()

    # Report results
    print(f"\nProcessed {total_processed} total examples")
    print("\nCollection results:")
    for domain in DOMAIN_RATIOS.keys():
        n_short = len(collected[domain]['short'])
        n_long = len(collected[domain]['long'])
        target = target_per_domain[domain]
        print(f"  {domain}: {n_short} short + {n_long} long = {n_short + n_long} / {target} target")

    return collected


def merge_and_create_df(collected, target_per_domain, seed=42):
    """Merge short and long examples, preferring short."""
    import random
    rng = random.Random(seed)

    all_examples = []

    for domain in DOMAIN_RATIOS.keys():
        target = target_per_domain[domain]
        short_examples = collected[domain]['short']
        long_examples = collected[domain]['long']

        # Shuffle long pool to avoid positional bias
        long_shuffled = long_examples.copy()
        rng.shuffle(long_shuffled)

        # Take all short, supplement with shuffled long
        domain_examples = short_examples.copy()
        needed = target - len(domain_examples)
        if needed > 0 and long_shuffled:
            domain_examples.extend(long_shuffled[:needed])

        # Mark which are long (for truncation later)
        for i, ex in enumerate(domain_examples):
            ex['_needs_truncation'] = i >= len(short_examples)

        all_examples.extend(domain_examples)
        print(f"{domain}: using {len(short_examples)} short + {min(needed, len(long_examples))} long")

    return pd.DataFrame(all_examples)


def save_jsonl(df, output_path):
    """Save dataframe to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    examples = []
    for _, row in df.iterrows():
        example = {
            'domain': row['domain'],
            'source': row['source'],
            'conversations': row['conversations'],
            'needs_truncation': row.get('_needs_truncation', False),
        }
        if 'difficulty' in row and pd.notna(row.get('difficulty')):
            example['difficulty'] = row['difficulty']
        examples.append(example)

    with open(output_path, 'w') as f:
        for example in examples:
            json.dump(example, f)
            f.write('\n')


def save_train_val_splits(df, base_path, n_train, n_val, seed):
    """Split and save train/val datasets."""
    total_needed = n_train + n_val
    if len(df) < total_needed:
        print(f"Warning: Only {len(df)} examples, need {total_needed}")
        ratio = n_train / total_needed
        n_train = int(len(df) * ratio)
        n_val = len(df) - n_train

    # Shuffle and split
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train:n_train + n_val]

    train_path = f"{base_path}_train.jsonl"
    val_path = f"{base_path}_val.jsonl"

    save_jsonl(train_df, train_path)
    save_jsonl(val_df, val_path)

    # Report truncation stats
    for name, split_df in [('train', train_df), ('val', val_df)]:
        n_trunc = split_df['_needs_truncation'].sum() if '_needs_truncation' in split_df else 0
        print(f"Saved {len(split_df)} {name} examples ({n_trunc} need truncation) to {name}_path")

    print(f"Train: {train_path}")
    print(f"Val: {val_path}")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Filter OpenThoughts with stratified sampling")
    parser.add_argument("--n_train", type=int, required=True, help="Number of training examples")
    parser.add_argument("--n_val", type=int, required=True, help="Number of validation examples")
    parser.add_argument("--soft_max_tokens", type=int, default=10000, help="Soft max tokens (prefer under, but allow over)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model for tokenizer")
    parser.add_argument("--format", type=str, default="deepseek", choices=["deepseek", "tokenizer"],
                        help="Format for tokenization: 'deepseek' (R1 format) or 'tokenizer' (chat template)")
    parser.add_argument("--output", type=str, help="Custom output base path")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Shuffle buffer size")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    total = args.n_train + args.n_val

    # Calculate target per domain based on ratios
    target_per_domain = {
        domain: int(total * ratio) + 1  # +1 buffer for rounding
        for domain, ratio in DOMAIN_RATIOS.items()
    }

    print(f"=== Stratified OpenThoughts Filtering ===")
    print(f"Total needed: {total} ({args.n_train} train, {args.n_val} val)")
    print(f"Domain ratios: {DOMAIN_RATIOS}")
    print(f"Targets: {target_per_domain}")
    print(f"Soft max tokens: {args.soft_max_tokens}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)

    # Collect examples
    collected = stream_and_collect_by_domain(
        target_per_domain=target_per_domain,
        soft_max_tokens=args.soft_max_tokens,
        tokenizer=tokenizer,
        format=args.format,
        buffer_size=args.buffer_size,
        seed=args.seed,
    )

    # Merge and create dataframe
    df = merge_and_create_df(collected, target_per_domain, seed=args.seed)
    print(f"\nTotal examples: {len(df)}")

    # Generate output path
    if args.output:
        base_path = args.output.replace('.jsonl', '').replace('_train', '').replace('_val', '')
    else:
        base_path = f"/nlp/scr/nathu/sparse-adaptation/data/openthoughts/stratified_n{total}_t{args.soft_max_tokens}_s{args.seed}"

    # Save splits
    save_train_val_splits(df, base_path, args.n_train, args.n_val, args.seed)


if __name__ == "__main__":
    main()
