"""
Pack per-feature JSON files into circuit-tracer binary format.

Converts a directory of {cantor_id}.json files into layer_*.bin files
with a compressed index, compatible with the circuit-tracer frontend.
See /nlp/u/nathu/circuit-tracer/docs/packed_feature_spec.md for format details.

Usage:
    python -m analysis.features.pack_features \
        --feature_dir /nlp/scr/nathu/sparse-adaptation/circuit_tracing/r1_distil_7b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_lr8e-04_bs1_2025-12-29_1408/features \
        --output_dir /nlp/scr/nathu/sparse-adaptation/circuit_tracing/r1_distil_7b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_lr8e-04_bs1_2025-12-29_1408/packed_features \
        --n_layers 28 \
        --n_features 8192

and to upload to hf:
    huggingface-cli upload nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4 path/to/packed_features features --repo-type model

"""

import argparse
import gzip
import json
import os
import struct
from typing import Any
from tqdm import tqdm


def cantor_pair(layer, feat_idx):
    return (layer + feat_idx) * (layer + feat_idx + 1) // 2 + feat_idx


def pack_feature(feature_dict: dict) -> bytes:
    """Pack a single feature JSON into the binary entry format."""
    json_bytes = json.dumps(feature_dict, separators=(',', ':')).encode('utf-8')
    compressed = gzip.compress(json_bytes)
    return struct.pack('<I', len(compressed)) + compressed


def pack_layer(feature_dir, bin_path, layer, n_features):
    """Pack all features for a single layer into a .bin file."""
    offsets = [0]
    n_found = 0

    with open(bin_path, "wb") as bin_f:
        for feat_idx in range(n_features):
            cantor_idx = cantor_pair(layer, feat_idx)
            json_path = os.path.join(feature_dir, f"{cantor_idx}.json")

            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    feature = json.load(f)
                bin_f.write(pack_feature(feature))
                n_found += 1

            offsets.append(bin_f.tell())

    return offsets, n_found


def main():
    parser = argparse.ArgumentParser(description="Pack feature JSONs into binary format")
    parser.add_argument("--feature_dir", required=True, help="Directory of {cantor_id}.json files")
    parser.add_argument("--output_dir", required=True, help="Where to write packed output")
    parser.add_argument("--n_layers", type=int, default=28)
    parser.add_argument("--n_features", type=int, default=8192)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    index: dict[str, Any] = {"version": "1.0", "format": "variable_chunks"}

    total_found = 0
    total_possible = args.n_layers * args.n_features

    for layer in tqdm(range(args.n_layers), desc="Packing layers"):
        bin_path = os.path.join(args.output_dir, f"layer_{layer}.bin")
        offsets, n_found = pack_layer(args.feature_dir, bin_path, layer, args.n_features)
        total_found += n_found

        index[str(layer)] = {
            "filename": f"layer_{layer}.bin",
            "offsets": offsets,
        }

        size_mb = os.path.getsize(bin_path) / 1e6
        n_dead = args.n_features - n_found
        tqdm.write(f"  Layer {layer}: {size_mb:.1f} MB, {n_found} features, {n_dead} dead")

    # Write compressed index
    index_path = os.path.join(args.output_dir, "index.json.gz")
    with gzip.open(index_path, "wt") as f:
        json.dump(index, f)

    total_dead = total_possible - total_found
    print(f"\nDone: {total_found}/{total_possible} features packed ({total_dead} dead)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
