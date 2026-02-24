#%%
"""
Batch attribution script for RelP models.

Takes a directory of prompt files, runs attribution on each, and saves
graph files for frontend visualization.

Prompt files should include the target token at the end (it will be dropped
to form the prompt, and used to verify the target).

Output structure (for running multiple models on same prompts):
    output_dir/
    ├── graph-metadata.json
    ├── {run_name}__{prompt_name}.json
    └── features/{scan}/...

Usage:
    python -m analysis.attribution.run_attribution \
        --checkpoint /path/to/model/folded_v2/ \
        --run_name r1_distil \
        --prompts_dir /path/to/prompts/ \
        --output_dir /path/to/output/

After running, start the frontend with:
    circuit-tracer start-server --graph_file_dir {output_dir} --port 8042
"""

import os
import argparse
from pathlib import Path

import torch

from analysis.attribution.relp_model import RelPReplacementModel
from analysis.attribution.attribute import attribute
from circuit_tracer.utils.create_graph_files import create_graph_files

#%%
def load_prompt_file(path: Path, tokenizer) -> tuple[list[int], int, str]:
    """
    Load a prompt file where the last token is the target.

    Returns:
        prompt_tokens: Token IDs for the prompt (excluding target)
        target_token: The target token ID
        prompt_str: Decoded prompt string
    """
    text = path.read_text()
    full_tokens = tokenizer.encode(text, add_special_tokens=False)
    prompt_tokens = full_tokens[:-1]
    target_token = full_tokens[-1]
    prompt_str = tokenizer.decode(prompt_tokens)
    return prompt_tokens, target_token, prompt_str


def load_prompts(prompts_dir: str, tokenizer) -> dict[str, dict]:
    """
    Load prompts from a directory of .txt files.

    Returns dict mapping slug to {tokens, target, text}.
    """
    prompts = {}
    prompts_path = Path(prompts_dir)

    if not prompts_path.exists():
        raise ValueError(f"Prompts directory does not exist: {prompts_dir}")

    for txt_file in sorted(prompts_path.glob("*.txt")):
        slug = txt_file.stem
        tokens, target, text = load_prompt_file(txt_file, tokenizer)
        prompts[slug] = {
            "tokens": tokens,
            "target": target,
            "text": text,
        }
        target_str = tokenizer.decode([target])
        print(f"  {slug}: {len(tokens)} tokens, target={target_str!r}")

    if not prompts:
        raise ValueError(f"No .txt files found in {prompts_dir}")

    return prompts


def run_attribution_for_prompt(
    prompt_tokens: list[int],
    slug: str,
    model: RelPReplacementModel,
    scan: str,
    output_dir: str,
    max_n_logits: int,
    batch_size: int,
    max_feature_nodes: int,
    node_threshold: float,
    edge_threshold: float,
):
    """Run attribution for a single prompt and save graph files."""
    import gc

    print(f"  Running attribution (batch_size={batch_size})...")
    raw_graph = attribute(
        prompt_tokens,
        model,
        max_n_logits=max_n_logits,
        max_feature_nodes=max_feature_nodes,
        batch_size=batch_size,
        verbose=True,
    )

    print(f"  Graph: {raw_graph.active_features.shape[0]} active, "
          f"{raw_graph.selected_features.shape[0]} selected")

    # BOS zeroing is done inline in attribute.py
    graph = raw_graph

    # Free up GPU memory before pruning (prune_graph needs to sort large edge matrix)
    graph.to("cpu")
    # Clear model's cached features and gradients
    for layer in model.model.model.layers:
        if hasattr(layer.mlp, 'cached_features'):
            layer.mlp.cached_features = None
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Save graph files - if pruning OOMs, save raw graph instead
    try:
        create_graph_files(
            graph_or_path=graph,
            slug=slug,
            scan=scan,
            output_path=output_dir,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
            raise
        # Save raw graph so we don't lose the expensive backward passes
        raw_graph_path = Path(output_dir) / f"{slug}_raw.pt"
        print(f"  OOM during pruning, saving raw graph to {raw_graph_path}")
        graph.to_pt(str(raw_graph_path))
        print(f"  Raw graph saved. Prune later with: create_graph_files('{raw_graph_path}', ...)")

    return graph


#%%
def main():
    parser = argparse.ArgumentParser(
        description="Run RelP attribution on multiple prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required args - explicit for accounting
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--run_name", type=str, required=True, help="Name for this run (used in output: {run_name}__{prompt}.json)")
    parser.add_argument("--prompts", type=str, required=True, help="Directory with .txt prompt files, or path to a single .txt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for graphs")

    # Optional: override scan name (defaults to run_name)
    parser.add_argument("--scan", type=str, default=None, help="Scan name for features (default: run_name)")

    # Attribution parameters
    parser.add_argument("--max_n_logits", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_feature_nodes", type=int, default=10000)
    parser.add_argument("--node_threshold", type=float, default=0.8)
    parser.add_argument("--edge_threshold", type=float, default=0.98)

    parser.add_argument("--device", type=str, default="cuda", help="Device (ignored if --device_map is set)")
    parser.add_argument("--device_map", type=str, default=None, help="Device map for multi-GPU. Use 'auto' to split layers across GPUs.")

    args = parser.parse_args()

    # Default scan to run_name
    if args.scan is None:
        args.scan = args.run_name

    # Print config
    print("=" * 60)
    print("RelP Attribution")
    print("=" * 60)
    print(f"Run name:    {args.run_name}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Prompts:     {args.prompts}")
    print(f"Output:      {args.output_dir}")
    print(f"Scan:        {args.scan}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model = RelPReplacementModel.from_pretrained(
        args.checkpoint,
        device=args.device,
        device_map=args.device_map,
        dtype=torch.bfloat16,
    )
    print(f"Loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} d_model, "
          f"{model.cfg.n_features} features")

    # Load prompts (single file or directory)
    print("\nLoading prompts...")
    prompts_path = Path(args.prompts)
    if prompts_path.is_file():
        slug = prompts_path.stem
        tokens, target, text = load_prompt_file(prompts_path, model.tokenizer)
        prompts = {slug: {"tokens": tokens, "target": target, "text": text}}
        target_str = model.tokenizer.decode([target])
        print(f"  {slug}: {len(tokens)} tokens, target={target_str!r}")
    else:
        prompts = load_prompts(args.prompts, model.tokenizer)
    prompt_items = list(prompts.items())
    print(f"Found {len(prompt_items)} prompt(s)")

    # Run attribution
    print("\n" + "=" * 60)
    print("Running Attribution")
    print("=" * 60)

    results = {}
    for i, (prompt_name, data) in enumerate(prompt_items, 1):
        slug = f"{args.run_name}__{prompt_name}"
        graph_path = Path(args.output_dir) / f"{slug}.json"

        # Skip if graph already exists
        if graph_path.exists():
            print(f"\n[{i}/{len(prompt_items)}] {slug} - SKIPPED (already exists)")
            results[prompt_name] = "skipped"
            continue

        print(f"\n[{i}/{len(prompt_items)}] {slug}")
        prompt_text: str = data['text']  # type: ignore[assignment]
        prompt_tokens: list[int] = data["tokens"]  # type: ignore[assignment]
        print(f"  Last 60 chars: ...{prompt_text[-60:]!r}")
        print(f"  Target: {model.tokenizer.decode([data['target']])!r}")

        try:
            graph = run_attribution_for_prompt(
                prompt_tokens=prompt_tokens,
                slug=slug,
                model=model,
                scan=args.scan,
                output_dir=args.output_dir,
                max_n_logits=args.max_n_logits,
                batch_size=args.batch_size,
                max_feature_nodes=args.max_feature_nodes,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
            )
            results[prompt_name] = "success"
            print(f"  Done: {slug}.json")
        except Exception as e:
            results[prompt_name] = f"error: {e}"
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

            # Clean up GPU memory after OOM to allow recovery
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                print("  Attempting CUDA memory cleanup...")
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    success = sum(1 for v in results.values() if v == "success")
    skipped = sum(1 for v in results.values() if v == "skipped")
    errors = len(prompt_items) - success - skipped
    print(f"Success: {success}, Skipped: {skipped}, Errors: {errors} (total: {len(prompt_items)})")

    print(f"\nTo view graphs run:")
    print(f"  circuit-tracer start-server --graph_file_dir {args.output_dir}")


#%%
if __name__ == "__main__":
    main()
