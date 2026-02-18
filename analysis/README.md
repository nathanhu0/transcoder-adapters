# Analysis

Tools for interpreting transcoder adapter features: collecting activation examples, classifying features, auto-interpretability, and generating attribution graphs viewable in circuit-tracer.

## Setup

Install dependencies (from repo root):

```bash
pip install -r requirements.txt
```

Feature classification and auto-interp require an OpenAI API key:

```bash
cp .env.example .env
# edit .env with your OPENAI_API_KEY
```

## Pipeline Overview

The main entry point is **collecting activating text** — this produces per-feature activation examples in an expanded JSON format that all other analysis steps consume.

After collection, three independent analyses can be run:
- **Classification** — LLM-judge categorization of features
- **Auto-interp** — automated feature descriptions with detection evaluation
- **Attribution** — graph-based analysis of feature interactions (requires additional steps: packing features, uploading to HF, and specifying prompt text)

```
analysis/
├── features/
│   ├── collect_feature_activations.py  # Step 1: collect activating text
│   ├── collect_neuron_activations.py   # Same, but for MLP neurons (baseline)
│   ├── classify_features.py            # LLM-judge classification
│   ├── auto_interp.py                  # Auto-interp with detection evaluation
│   └── pack_features.py                # Pack into circuit-tracer binary format
├── attribution/
│   ├── run_attribution.py              # CLI: generate attribution graphs
│   ├── attribute.py                    # Core RelP attribution algorithm
│   ├── relp_model.py                   # Model wrapper for attribution
│   └── relp_context.py                 # Forward/backward caching for RelP
```

## Step 1: Collect activating text

Runs the model on validation data and collects top-activating examples, logit lens, and activation statistics for each feature. This is a prerequisite for all other analysis.

```bash
python -m analysis.features.collect_feature_activations \
    --model_path nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4 \
    --val_data hf://nathu0/transcoder-adapters-openthoughts3-stratified-55k/data/val.jsonl \
    --output_dir ./feature_data
```

Outputs:
- `features/{cantor_id}.json` — per-feature activation examples + logit lens
- `feature_metadata.json` — activation frequencies, domain/region breakdowns

## Step 2: Feature analysis (independent)

These steps can be run in any order after step 1.

### Classify features

LLM-judge classification into categories (language, domain, reasoning, uninterpretable).

```bash
python -m analysis.features.classify_features \
    --input_dir ./feature_data \
    --output feature_classifications.json \
    --n_per_layer 250
```

### Auto-interpretability (optional)

Generates feature descriptions and evaluates with a detection task.

```bash
python -m analysis.features.auto_interp \
    --input_dir ./feature_data \
    --data_path /path/to/openthoughts_val.jsonl \
    --output autointerp.json \
    --n_per_layer 100
```

## Step 3: Attribution graphs

Attribution graphs show how features influence each other and the model's output for a specific prompt. The graphs are viewable in the [circuit-tracer](https://github.com/safety-research/circuit-tracer) frontend, which loads feature data from HuggingFace.

**To generate graphs on new text** (for a model whose features are already on HF), just write your prompts and run attribution:

```bash
python -m analysis.attribution.run_attribution \
    --checkpoint nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4 \
    --run_name r1_l0_1p4 \
    --scan nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4 \
    --prompts /path/to/prompts/ \
    --output_dir ./graph_files
```

- `--checkpoint` can be an HF repo ID or a local path
- `--scan` must be an HF repo ID — the circuit-tracer frontend loads feature activation data from HuggingFace at browse time
- `--prompts` can be a directory of `.txt` files or a single `.txt` file. Attribution is computed for the prediction of the final token — the last token in the file is the target being predicted and is included for readability but gets dropped from the prompt.
- `--batch_size` controls memory usage (default 16); lower if you hit OOM
- Already-computed graphs are skipped on re-run

**To analyze a new model**, you first need to collect, pack, and upload features before running attribution:

1. **Collect** activating text (step 1 above)
2. **Pack** into circuit-tracer binary format:
   ```bash
   python -m analysis.features.pack_features \
       --feature_dir ./feature_data/features \
       --output_dir ./packed_features \
       --n_layers 28 \
       --n_features 8192
   ```
3. **Upload** to HuggingFace:
   ```bash
   huggingface-cli upload <your-hf-repo> \
       ./packed_features features --repo-type model
   ```
4. Then run attribution as above, setting `--checkpoint` and `--scan` to your HF repo ID.

### View in browser

Start the circuit-tracer frontend:

```bash
circuit-tracer start-server --graph_file_dir ./graph_files --port 8042
```

Open `http://localhost:8042`. Click any feature node to load its activation examples from HuggingFace.

## Pre-built Data

| Resource | Location |
|----------|----------|
| Packed features (l1w0.001) | `features/` folder in [model repo](https://huggingface.co/nathu0/transcoder-adapters-R1-Distill-Qwen-7B-l1w0.001-l0-1.4) |
| Feature classifications | `feature_classifications.json` in model repo |
| Pre-built attribution graphs | TODO |
