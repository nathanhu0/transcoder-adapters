"""
One-time script to copy graph files and patch them so the circuit-tracer
frontend fetches features from HuggingFace instead of local paths.

Copies the entire graph directory first, then patches the scan field in
each graph JSON. The frontend resolves scan to:
  https://huggingface.co/{scan}/resolve/main/features/{path}

Usage:
    python misc_scripts/prepare_graphs_for_hf.py \
        --graph_dir /nlp/scr/nathu/sparse-adaptation/circuit_tracing_final \
        --output_dir ./graph_files \
        --scan nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4
"""

import argparse
import json
import os
import shutil
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", required=True, help="Directory with original graph JSONs")
    parser.add_argument("--output_dir", required=True, help="Where to write patched copy")
    parser.add_argument("--scan", required=True, help="HF repo ID for features")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        print(f"Error: output_dir already exists: {args.output_dir}")
        print("Remove it first to avoid mixing old and new files.")
        return

    # Step 1: Copy everything except features/ (which may be a symlink to
    # a large directory of raw JSONs — we don't need it since features will
    # be loaded from HF)
    print(f"Copying {args.graph_dir} -> {args.output_dir} (skipping features/)")
    shutil.copytree(args.graph_dir, args.output_dir,
                    ignore=shutil.ignore_patterns("features"))
    print("  Done.")

    # Step 2: Patch scan field in graph JSONs
    graph_files = glob.glob(os.path.join(args.output_dir, "*.json"))
    patched = 0
    for path in sorted(graph_files):
        filename = os.path.basename(path)
        if filename == "graph-metadata.json":
            continue

        with open(path) as f:
            data = json.load(f)

        if "metadata" in data and "scan" in data["metadata"]:
            old_scan = data["metadata"]["scan"]
            data["metadata"]["scan"] = args.scan
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  {filename}: scan '{old_scan}' -> '{args.scan}'")
            patched += 1
        else:
            print(f"  {filename}: no metadata.scan, skipped")

    print(f"\nDone. Copied directory and patched {patched} graph files.")


if __name__ == "__main__":
    main()
