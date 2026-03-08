#%% Imports and setup
"""Create a hybrid model: target model with base model MLP weights swapped in.

This is a baseline for transcoder adapter experiments. The hybrid model uses:
- Attention, embeddings, layernorms from the target (reference/reasoning) model
- MLP gate_proj, up_proj, down_proj from the base model

Tokenizer and config are copied from the target model since that's what
defines the chat template, special tokens, and vocabulary.

Usage:
    python misc_scripts/make_hybrid_model.py \
        --base_model Qwen/Qwen2.5-32B \
        --target_model Qwen/QwQ-32B \
        --output_dir /path/to/hybrid_checkpoint
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#%% Parse arguments
parser = argparse.ArgumentParser(description="Create hybrid model (target + base MLP)")
parser.add_argument("--base_model", type=str, required=True,
                    help="Base model path (MLP donor)")
parser.add_argument("--target_model", type=str, required=True,
                    help="Target/reference model path (attention/embed/layernorm donor)")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Where to save the hybrid checkpoint")
parser.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Model dtype (default: bfloat16)")
args = parser.parse_args()

torch_dtype = getattr(torch, args.dtype)

#%% Load target model (this becomes the hybrid — we'll swap MLP weights in-place)
print(f"Loading target model: {args.target_model}")
model = AutoModelForCausalLM.from_pretrained(
    args.target_model,
    torch_dtype=torch_dtype,
    device_map="cpu",
    trust_remote_code=True,
)

#%% Load base model (MLP weight donor)
print(f"Loading base model: {args.base_model}")
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch_dtype,
    device_map="cpu",
    trust_remote_code=True,
)

#%% Verify layer counts match
n_layers_target = len(model.model.layers)
n_layers_base = len(base_model.model.layers)
assert n_layers_target == n_layers_base, (
    f"Layer count mismatch: target has {n_layers_target}, base has {n_layers_base}"
)
print(f"Both models have {n_layers_target} layers")

#%% Swap MLP weights from base into target
print("Swapping MLP weights (gate_proj, up_proj, down_proj)...")
params_swapped = 0
for i in range(n_layers_target):
    target_mlp = model.model.layers[i].mlp
    base_mlp = base_model.model.layers[i].mlp

    for proj in ("gate_proj", "up_proj", "down_proj"):
        target_w = getattr(target_mlp, proj).weight
        base_w = getattr(base_mlp, proj).weight
        assert target_w.shape == base_w.shape, f"Layer {i} {proj} shape mismatch"
        target_w.data.copy_(base_w.data)
        params_swapped += target_w.numel()

print(f"MLP weights swapped: {3 * n_layers_target} projections, {params_swapped:,} params")

#%% Free base model memory
del base_model
import gc
gc.collect()

#%% Load tokenizer from target model
print(f"Loading tokenizer from target model: {args.target_model}")
tokenizer = AutoTokenizer.from_pretrained(
    args.target_model, trust_remote_code=True, use_fast=True,
)

#%% Save hybrid model with target's config and tokenizer
import os
os.makedirs(args.output_dir, exist_ok=True)

print(f"Saving hybrid model to: {args.output_dir}")
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

#%% Print summary
print(f"\nHybrid model saved to: {args.output_dir}")
print(f"  Target (attn/embed/norm): {args.target_model}")
print(f"  Base (MLP): {args.base_model}")
print(f"  Tokenizer: from target model")
print(f"  eos_token_id: {model.config.eos_token_id}")
print(f"  vocab_size: {model.config.vocab_size}")
print(f"  num_layers: {n_layers_target}")
