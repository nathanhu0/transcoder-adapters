#!/usr/bin/env python3
"""Training script for bridging experiments.

This script trains sparse adapters (e.g., transcoder) with bridging losses
that encourage layer-wise compatibility with a reference model.
"""

import os
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm import tqdm
import argparse
import wandb

from training.config import load_config, ExperimentConfig, _finalize_config
from training.dataset.PredefinedDataset import PredefinedDataset
from training.forward_utils import forward_mixed, sample_cutoffs
from training.losses import compute_kl_loss, compute_lm_loss, compute_nmse_loss
from models import get_transcoder_classes

DEBUG_MODE_EARLY_EXIT_STEPS = 50

def move_batch_to(device, batch):
    """Move batch tensors to device."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def setup_models_bridging(config: ExperimentConfig):
    """Load transcoder model and frozen reference model (bridging mode).

    Model assembly:
      1. Load Qwen2ForCausalLMWithTranscoder via from_pretrained. This loads all
         standard Qwen2 weights (attention, embeddings, MLP gate/up/down) from the
         checkpoint. Transcoder parameters (transcoder_enc, transcoder_dec) are NOT
         in the checkpoint and stay at their __init__ values (dec=zeros, so zero
         initial contribution).
      2. For "target" backbone: the checkpoint is the reference model (DeepSeek R1
         Distill), so we additionally swap in the base model's MLP weights
         (gate_proj, up_proj, down_proj). This gives us: reference attention/embed
         + base MLP + fresh transcoder.
      3. After training, model.save_pretrained() saves everything (including trained
         transcoder weights) as a single checkpoint — no conversion step needed.
    """
    bridging_config = config.bridging
    assert bridging_config is not None
    backbone = getattr(bridging_config, 'backbone', 'base')
    tc_config = config.transcoder
    assert tc_config is not None

    assert config.model_arch is not None, "model_arch must be set (auto-detected or explicit)"
    ConfigWithTranscoder, ModelWithTranscoder = get_transcoder_classes(config.model_arch)

    # Load tokenizer from reference model (the model we're distilling toward)
    tokenizer_path = bridging_config.reference_model_path if bridging_config else config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build HF config with transcoder params
    hf_config = ConfigWithTranscoder.from_pretrained(
        config.model_name,
        transcoder_n_features=tc_config.n_features,
        transcoder_dec_bias=tc_config.dec_bias,
    )

    # Load transcoder model. from_pretrained loads standard weights from the
    # checkpoint; transcoder_enc/dec are not in the checkpoint and stay at __init__
    # values (dec=zeros → zero initial contribution).
    if backbone == "target":
        # Target backbone: load reference model (attn/embed/layernorm from reference),
        # then swap in base model's MLP weights. Result: reference attn + base MLP + fresh transcoder.
        print(f"Loading reference model as backbone: {bridging_config.reference_model_path}")
        model = ModelWithTranscoder.from_pretrained(
            bridging_config.reference_model_path,
            config=hf_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Swapping in base model MLP weights from: {config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        for adapter_mlp, base_layer in zip(model._transcoder_mlps(), base_model.model.layers): # pyright: ignore[reportCallIssue]
            base_mlp = base_layer.mlp  # type: ignore[union-attr]
            device = adapter_mlp.gate_proj.weight.device
            adapter_mlp.gate_proj.weight.data.copy_(base_mlp.gate_proj.weight.data.to(device))
            adapter_mlp.up_proj.weight.data.copy_(base_mlp.up_proj.weight.data.to(device))
            adapter_mlp.down_proj.weight.data.copy_(base_mlp.down_proj.weight.data.to(device))
        del base_model
        print("MLP weights swapped")
    else:
        # Base backbone: all non-transcoder weights from base model directly.
        print(f"Loading base model: {config.model_name}")
        model = ModelWithTranscoder.from_pretrained(
            config.model_name,
            config=hf_config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    # Re-initialize transcoder weights. from_pretrained with device_map="auto" creates
    # meta tensors first, so our __init__ zero-initialization of dec is overwritten by
    # HF's default _init_weights (normal distribution). This restores dec=zeros for
    # zero initial transcoder contribution.
    for mlp in model._transcoder_mlps(): # pyright: ignore[reportCallIssue]
        mlp._init_transcoder_weights()

    # Freeze everything except transcoder parameters
    for name, param in model.named_parameters():
        param.requires_grad = "transcoder" in name
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {n_trainable:,} params, Frozen: {n_frozen:,} params")

    # Load reference model (frozen)
    print(f"Loading reference model: {bridging_config.reference_model_path}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        bridging_config.reference_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("Reference model loaded and frozen")

    return model, ref_model, tokenizer


def setup_models_direct(config: ExperimentConfig):
    """Load base model with transcoders for direct fine-tuning (no reference model).

    Adds special tokens (e.g. <think>, </think>) to the base tokenizer and copies
    their embeddings from the reference model so they have meaningful representations
    despite being frozen.
    """
    direct_config = config.direct
    assert direct_config is not None
    tc_config = config.transcoder
    assert tc_config is not None

    assert config.model_arch is not None, "model_arch must be set (auto-detected or explicit)"
    ConfigWithTranscoder, ModelWithTranscoder = get_transcoder_classes(config.model_arch)

    # Load base model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens that the base tokenizer doesn't have
    if direct_config.copied_tokens:
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": direct_config.copied_tokens,
        })
        print(f"Added {num_added} special tokens: {direct_config.copied_tokens}")

    # Load base model with transcoders
    hf_config = ConfigWithTranscoder.from_pretrained(
        config.model_name,
        transcoder_n_features=tc_config.n_features,
        transcoder_dec_bias=tc_config.dec_bias,
    )
    print(f"Loading base model: {config.model_name}")
    model = ModelWithTranscoder.from_pretrained(
        config.model_name,
        config=hf_config,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize embeddings for newly added tokens
    if direct_config.copied_tokens:
        model.resize_token_embeddings(len(tokenizer))

        # Copy embeddings from reference model for the new tokens
        print(f"Copying token embeddings from: {direct_config.reference_model_path}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            direct_config.reference_model_path,
            dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        ref_tokenizer = AutoTokenizer.from_pretrained(
            direct_config.reference_model_path, trust_remote_code=True,
        )

        for token_str in direct_config.copied_tokens:
            # Get token ID in each tokenizer
            new_id = tokenizer.convert_tokens_to_ids(token_str)
            ref_id = ref_tokenizer.convert_tokens_to_ids(token_str)
            if ref_id == ref_tokenizer.unk_token_id:
                print(f"  WARNING: '{token_str}' not found in reference tokenizer, skipping")
                continue

            # Copy embed_tokens
            embed_device = model.model.embed_tokens.weight.device
            model.model.embed_tokens.weight.data[new_id] = (
                ref_model.model.embed_tokens.weight.data[ref_id].to(embed_device)
            )
            # Copy lm_head
            head_device = model.lm_head.weight.device
            model.lm_head.weight.data[new_id] = (
                ref_model.lm_head.weight.data[ref_id].to(head_device)
            )
            print(f"  Copied '{token_str}': ref[{ref_id}] -> base[{new_id}]")

        del ref_model, ref_tokenizer
        print("Token embeddings copied")

    # Re-initialize transcoder weights (same reason as bridging)
    for mlp in model._transcoder_mlps(): # pyright: ignore[reportCallIssue]
        mlp._init_transcoder_weights()

    # Freeze everything except transcoder parameters
    for name, param in model.named_parameters():
        param.requires_grad = "transcoder" in name
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {n_trainable:,} params, Frozen: {n_frozen:,} params")

    return model, None, tokenizer


def setup_data(config: ExperimentConfig, tokenizer):
    """Setup dataset and dataloader."""

    dataset_loader = PredefinedDataset(
        dataset_type=config.dataset_type,
        tokenizer=tokenizer,
        length_excession_behavior=config.length_excession_behavior,
        loss_on_prompt=config.loss_on_prompt,
        dataset_specific_config=config.dataset,
    )
    train_dataset, dataloaders = dataset_loader.load_dataset_and_dataloaders()

    return train_dataset, dataloaders["train"], dataloaders.get("val", None)


def setup_training(config: ExperimentConfig, model, dataset):
    """Setup optimizer and scheduler."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=0.0
    )

    steps_per_epoch = len(dataset) // config.batch_size
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return optimizer, scheduler, total_steps, warmup_steps


def train_step_direct(
    model,
    batch: dict,
    config: ExperimentConfig,
    gradient_accumulation_steps: int,
) -> dict[str, float]:
    """Single training step for direct fine-tuning (LM loss + sparsity only)."""
    assert config.transcoder is not None
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    l1_weight = config.transcoder.l1_weight or 0.0

    # Forward pass with feature caching (for L1 loss + stats)
    model.set_cache_features(True)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    model.set_cache_features(False)

    # Losses
    lm_loss = compute_lm_loss(logits, labels)
    raw_sparsity = model.collect_sparsity_loss()
    weighted_sparsity = l1_weight * raw_sparsity

    # Backward
    total_loss = (lm_loss + weighted_sparsity) / gradient_accumulation_steps
    total_loss.backward()

    metrics = {
        "train/lm_loss": lm_loss.item(),
        "train/sparsity": raw_sparsity.item(),
        "train/total_loss": lm_loss.item() + weighted_sparsity.item(),
    }
    return metrics


def train_step_bridging(
    model,
    ref_model,
    batch: dict,
    config: ExperimentConfig,
    global_step: int,
    total_steps: int,
    gradient_accumulation_steps: int,
) -> dict[str, float]:
    """Single training step with bridging loss using memory-efficient forward_mixed.

    This function handles its own backward() calls to free computation graphs
    immediately after each loss component, avoiding memory buildup.

    Returns:
        Dict of metrics (losses are already backward'd, no tensor returned)
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    bridging_config = config.bridging
    assert bridging_config is not None
    assert config.transcoder is not None
    n_layers = len(model.model.layers)
    l1_weight = config.transcoder.l1_weight or 0.0

    # 1. Main forward pass with feature caching (for L1 loss + stats)
    model.set_cache_features(True)
    logits_adapt = model(input_ids=input_ids, attention_mask=attention_mask).logits
    model.set_cache_features(False)

    # Collect sparsity loss (sum of L1 across layers, must be while graph alive)
    raw_sparsity = model.collect_sparsity_loss()
    sparsity_loss = l1_weight * raw_sparsity

    # 2. Clean ref forward (for logits_ref if using KL loss)
    with torch.no_grad():
        logits_ref = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Track metrics
    metrics = {}
    total_loss_value = 0.0

    # 3. Adapt-only loss + sparsity loss (backward together)
    # Compute the training loss with grad, and the other one without grad for logging
    adapt_only_loss = torch.tensor(0.0, device=logits_adapt.device)
    if bridging_config.loss_type == "kl":
        kl_to_ref = compute_kl_loss(logits_adapt, logits_ref, labels)
        with torch.no_grad():
            lm_loss = compute_lm_loss(logits_adapt, labels)
        if bridging_config.always_include_adapt_only:
            adapt_only_loss = kl_to_ref
    else:
        lm_loss = compute_lm_loss(logits_adapt, labels)
        with torch.no_grad():
            kl_to_ref = compute_kl_loss(logits_adapt, logits_ref, labels)
        if bridging_config.always_include_adapt_only:
            adapt_only_loss = lm_loss

    metrics["train/kl_to_ref"] = kl_to_ref.item()
    metrics["train/lm_loss"] = lm_loss.item()
    if bridging_config.always_include_adapt_only:
        metrics["bridging/adapt_only"] = adapt_only_loss.item()
        total_loss_value += bridging_config.lambda_adapt * adapt_only_loss.item()

    # Combined loss for first backward (adapt_only + sparsity)
    first_loss = bridging_config.lambda_adapt * adapt_only_loss + sparsity_loss
    if first_loss.requires_grad:
        scaled_first = first_loss / gradient_accumulation_steps
        scaled_first.backward()

    metrics["train/sparsity"] = raw_sparsity.item()

    # Free logits_adapt to save memory
    del logits_adapt

    # 4. NMSE loss (if enabled) - backward separately
    if bridging_config.lambda_nmse > 0:
        nmse_loss, nmse_layerwise = compute_nmse_loss(
            model, ref_model, input_ids, attention_mask, return_layerwise=True
        )
        scaled_nmse = (bridging_config.lambda_nmse * nmse_loss) / gradient_accumulation_steps
        scaled_nmse.backward()
        metrics["bridging/nmse"] = nmse_loss.item()
        total_loss_value += bridging_config.lambda_nmse * nmse_loss.item()
        # Log per-layer NMSE
        for layer_idx, nmse_val in nmse_layerwise.items(): # type: ignore
            metrics[f"bridging_layerwise/nmse_layer_{layer_idx}"] = nmse_val
        del nmse_loss

    # 5. Bridging losses at sampled cutoffs (each backward immediately)
    bridge_loss_total = 0.0
    if bridging_config.n_cutoffs > 0:
        cutoffs = sample_cutoffs(n_layers, bridging_config.n_cutoffs, bridging_config.sampling)

        for k in cutoffs:
            # adapt->ref: adapter layers 0..k-1, ref layers k..L
            # Skip k=0 (no adapter layers used, no gradients)
            if k > 0:
                logits_a2r = forward_mixed(model, ref_model, input_ids, attention_mask, switch_layer=k)
                # Compute training loss with grad, other for logging only
                if bridging_config.loss_type == "kl":
                    loss_a2r = compute_kl_loss(logits_a2r, logits_ref, labels)
                    with torch.no_grad():
                        loss_a2r_lm = compute_lm_loss(logits_a2r, labels)
                    metrics[f"bridging_layerwise/a2r_kl_layer_{k}"] = loss_a2r.item()
                    metrics[f"bridging_layerwise/a2r_lm_layer_{k}"] = loss_a2r_lm.item()
                else:
                    loss_a2r = compute_lm_loss(logits_a2r, labels)
                    with torch.no_grad():
                        loss_a2r_kl = compute_kl_loss(logits_a2r, logits_ref, labels)
                    metrics[f"bridging_layerwise/a2r_lm_layer_{k}"] = loss_a2r.item()
                    metrics[f"bridging_layerwise/a2r_kl_layer_{k}"] = loss_a2r_kl.item()
                scaled_a2r = (bridging_config.lambda_bridge * loss_a2r) / (len(cutoffs) * gradient_accumulation_steps)
                scaled_a2r.backward()
                bridge_loss_total += loss_a2r.item()
                del logits_a2r, loss_a2r

            # ref->adapt: ref layers 0..k-1, adapter layers k..L
            # Skip k=n_layers (no adapter layers used, no gradients)
            if k < n_layers:
                logits_r2a = forward_mixed(ref_model, model, input_ids, attention_mask, switch_layer=k)
                # Compute training loss with grad, other for logging only
                if bridging_config.loss_type == "kl":
                    loss_r2a = compute_kl_loss(logits_r2a, logits_ref, labels)
                    with torch.no_grad():
                        loss_r2a_lm = compute_lm_loss(logits_r2a, labels)
                    metrics[f"bridging_layerwise/r2a_kl_layer_{k}"] = loss_r2a.item()
                    metrics[f"bridging_layerwise/r2a_lm_layer_{k}"] = loss_r2a_lm.item()
                else:
                    loss_r2a = compute_lm_loss(logits_r2a, labels)
                    with torch.no_grad():
                        loss_r2a_kl = compute_kl_loss(logits_r2a, logits_ref, labels)
                    metrics[f"bridging_layerwise/r2a_lm_layer_{k}"] = loss_r2a.item()
                    metrics[f"bridging_layerwise/r2a_kl_layer_{k}"] = loss_r2a_kl.item()
                scaled_r2a = (bridging_config.lambda_bridge * loss_r2a) / (len(cutoffs) * gradient_accumulation_steps)
                scaled_r2a.backward()
                bridge_loss_total += loss_r2a.item()
                del logits_r2a, loss_r2a

        # Average bridge loss for logging
        metrics["bridging/bridge"] = bridge_loss_total / (2 * len(cutoffs))
        total_loss_value += bridging_config.lambda_bridge * metrics["bridging/bridge"]

    metrics["bridging/total"] = total_loss_value
    metrics["train/total_loss"] = total_loss_value

    return metrics


def train_epoch(
    model,
    ref_model,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    config: ExperimentConfig,
    epoch: int,
    starting_step: int,
    total_steps: int,
    total_samples_seen: int,
    val_dataloader=None,
):
    """Train for one epoch."""
    model.train()
    batch_losses = []
    global_step = starting_step
    accumulation_step = 0
    current_batch_losses = []
    current_metrics = {}
    samples_seen = total_samples_seen

    assert config.micro_batch_size is not None
    gradient_accumulation_steps = config.batch_size // config.micro_batch_size # Note: not rly doing gradient accumulation anymore, see PredefinedDataset._make_dataloader comment.
    gradient_accumulation_steps = 1

    embed_device = model.get_input_embeddings().weight.device
    epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

    for step, batch in enumerate(epoch_pbar):
        batch = move_batch_to(embed_device, batch)

        # Forward + backward (handled inside train_step for memory efficiency)
        if config.direct:
            step_metrics = train_step_direct(
                model, batch, config, gradient_accumulation_steps
            )
        else:
            step_metrics = train_step_bridging(
                model, ref_model, batch, config,
                global_step, total_steps,
                gradient_accumulation_steps
            )

        accumulation_step += 1
        current_batch_losses.append(step_metrics.get("train/total_loss", 0.0))
        samples_seen += config.micro_batch_size

        # Accumulate metrics
        for k, v in step_metrics.items():
            if k not in current_metrics:
                current_metrics[k] = []
            current_metrics[k].append(v)

        # Accumulate truncation stats from batch
        if "truncated" in batch:
            if "data/truncated" not in current_metrics:
                current_metrics["data/truncated"] = []
                current_metrics["data/original_length"] = []
            current_metrics["data/truncated"].extend(batch["truncated"])
            current_metrics["data/original_length"].extend(batch["original_length"])

        # Optimizer step
        if accumulation_step % gradient_accumulation_steps == 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.gradient_clip_norm
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Average losses for this batch
            avg_batch_loss = sum(current_batch_losses) / len(current_batch_losses)
            batch_losses.append(avg_batch_loss)

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            epoch_pbar.set_postfix({
                'loss': f"{avg_batch_loss:.4f}",
                'lr': f"{current_lr:.2e}",
                'grad_norm': f"{total_norm:.3f}",
            })

            # Collect transcoder stats for logging
            for stat_name, stat_value in model.collect_transcoder_stats().items():
                current_metrics[stat_name] = [stat_value]

            # Log to WandB
            if config.use_wandb:
                log_dict = {
                    "train/total_loss": avg_batch_loss,
                    "train/learning_rate": current_lr,
                    "train/gradient_norm": total_norm.item(),
                    "train/epoch": epoch,
                    "train/step": global_step,
                    "train/samples_seen": samples_seen,
                }
                # Add averaged metrics (includes data/truncated as fraction, data/original_length as avg)
                for k, values in current_metrics.items():
                    log_dict[k] = sum(values) / len(values) if values else 0.0

                wandb.log(log_dict, step=global_step)

            # Reset accumulators
            current_batch_losses = []
            current_metrics = {}

            model.clear_cached_stats()

            # Run validation
            if val_dataloader is not None and global_step % config.val_frequency == 0:
                if config.direct:
                    val_metrics = validate_direct(model, val_dataloader, config)
                    val_metrics["val/epoch"] = epoch
                    val_metrics["val/step"] = global_step
                    val_metrics["val/samples_seen"] = samples_seen
                    print(f"  Val LM: {val_metrics['val/lm_loss']:.4f}")
                else:
                    val_metrics = validate_bridging(model, ref_model, val_dataloader, config)
                    val_metrics["val/epoch"] = epoch
                    val_metrics["val/step"] = global_step
                    val_metrics["val/samples_seen"] = samples_seen
                    print(f"  Val total: {val_metrics['val/total_loss']:.4f}, LM: {val_metrics['val/language_modeling_loss']:.4f}, KL: {val_metrics['val/kl_to_ref']:.4f}")
                if config.use_wandb:
                    wandb.log(val_metrics, step=global_step)

            # Run comprehensive layerwise validation (bridging only)
            if config.bridging and val_dataloader is not None and global_step % config.layerwise_val_frequency == 0:
                print("  Running layerwise validation...")
                layerwise_metrics = validate_layerwise(model, ref_model, val_dataloader, config)
                if config.use_wandb:
                    wandb.log(layerwise_metrics, step=global_step)

            # Save periodic checkpoint (overwrites previous latest)
            if config.save_checkpoints and global_step > 0 and global_step % config.checkpoint_frequency == 0:
                print(f"  Saving checkpoint at step {global_step}...")
                save_latest_checkpoint(model, tokenizer, config.output_dir, global_step)

            # Debug mode early exit
            if config.debug_mode and global_step >= DEBUG_MODE_EARLY_EXIT_STEPS:
                print(f"Debug mode: Breaking after {global_step} steps")
                break

        del batch

    avg_epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
    return avg_epoch_loss, global_step, samples_seen


@torch.no_grad()
def validate_direct(
    model,
    val_dataloader,
    config: ExperimentConfig,
    max_samples: int = 100,
) -> dict[str, float]:
    """Run validation for direct fine-tuning (LM loss + sparsity)."""
    model.eval()
    total_lm_loss = 0.0
    total_sparsity = 0.0
    num_samples = 0
    embed_device = model.get_input_embeddings().weight.device

    for batch in val_dataloader:
        if num_samples >= max_samples:
            break
        batch = move_batch_to(embed_device, batch)
        model.set_cache_features(True)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        model.set_cache_features(False)
        total_sparsity += model.collect_sparsity_loss().item()
        model.clear_cached_stats()
        total_lm_loss += compute_lm_loss(logits, batch["labels"]).item()
        num_samples += 1
        del batch

    model.train()
    return {
        "val/lm_loss": total_lm_loss / num_samples if num_samples > 0 else 0.0,
        "val/sparsity": total_sparsity / num_samples if num_samples > 0 else 0.0,
    }


@torch.no_grad()
def validate_bridging(
    model,
    ref_model,
    val_dataloader,
    config: ExperimentConfig,
) -> dict[str, float]:
    """Run validation on 100 samples (bridging mode)."""
    assert config.bridging is not None
    model.eval()

    total_lm_loss = 0.0
    total_kl_loss = 0.0
    total_sparsity_loss = 0.0
    total_loss = 0.0
    num_samples = 0
    max_samples = 100

    embed_device = model.get_input_embeddings().weight.device

    for batch in val_dataloader:
        if num_samples >= max_samples:
            break

        batch = move_batch_to(embed_device, batch)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Forward with feature caching for sparsity
        model.set_cache_features(True)
        logits_adapt = model(input_ids=input_ids, attention_mask=attention_mask).logits
        model.set_cache_features(False)
        raw_sparsity = model.collect_sparsity_loss().item()
        model.clear_cached_stats()
        total_sparsity_loss += raw_sparsity

        logits_ref = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Compute LM loss
        lm_loss = compute_lm_loss(logits_adapt, labels)
        total_lm_loss += lm_loss.item()

        # Compute KL loss (adapter -> ref)
        kl_loss = compute_kl_loss(logits_adapt, logits_ref, labels)
        total_kl_loss += kl_loss.item()

        # Total loss (using whatever loss_type is configured)
        if config.bridging.loss_type == "kl":
            task_loss = kl_loss.item()
        else:
            task_loss = lm_loss.item()
        total_loss += task_loss

        num_samples += 1
        del batch

    model.train()

    return {
        "val/total_loss": total_loss / num_samples if num_samples > 0 else 0.0,
        "val/language_modeling_loss": total_lm_loss / num_samples if num_samples > 0 else 0.0,
        "val/kl_to_ref": total_kl_loss / num_samples if num_samples > 0 else 0.0,
        "val/sparsity": total_sparsity_loss / num_samples if num_samples > 0 else 0.0,
    }


@torch.no_grad()
def validate_layerwise(
    model,
    ref_model,
    val_dataloader,
    config: ExperimentConfig,
    max_samples: int = 25,
) -> dict[str, float]:
    """Compute comprehensive layer-wise bridging and NMSE metrics on a small sample."""
    model.eval()

    n_layers = len(model.model.layers)
    embed_device = model.get_input_embeddings().weight.device

    # Accumulators for layer-wise metrics
    a2r_lm = {k: 0.0 for k in range(1, n_layers + 1)}
    a2r_kl = {k: 0.0 for k in range(1, n_layers + 1)}
    r2a_lm = {k: 0.0 for k in range(n_layers)}
    r2a_kl = {k: 0.0 for k in range(n_layers)}
    nmse_by_layer = {k: 0.0 for k in range(n_layers + 1)}
    num_samples = 0

    for batch in val_dataloader:
        if num_samples >= max_samples:
            break

        batch = move_batch_to(embed_device, batch)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Get ref logits for KL computation
        logits_ref = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Layer-wise bridging losses
        for k in range(n_layers + 1):
            if k > 0:  # a2r: adapter 0..k-1, ref k..L
                logits_a2r = forward_mixed(model, ref_model, input_ids, attention_mask, switch_layer=k)
                a2r_lm[k] += compute_lm_loss(logits_a2r, labels).item()
                a2r_kl[k] += compute_kl_loss(logits_a2r, logits_ref, labels).item()
                del logits_a2r

            if k < n_layers:  # r2a: ref 0..k-1, adapter k..L
                logits_r2a = forward_mixed(ref_model, model, input_ids, attention_mask, switch_layer=k)
                r2a_lm[k] += compute_lm_loss(logits_r2a, labels).item()
                r2a_kl[k] += compute_kl_loss(logits_r2a, logits_ref, labels).item()
                del logits_r2a

        # NMSE by layer
        _, nmse_layerwise = compute_nmse_loss(model, ref_model, input_ids, attention_mask, return_layerwise=True)
        assert isinstance(nmse_layerwise, dict)
        for layer_idx, val in nmse_layerwise.items():
            nmse_by_layer[layer_idx] += val

        num_samples += 1
        del batch, logits_ref

    model.train()

    # Build results
    results = {}
    for k in range(1, n_layers + 1):
        results[f"val_layerwise/a2r_lm_layer_{k}"] = a2r_lm[k] / num_samples
        results[f"val_layerwise/a2r_kl_layer_{k}"] = a2r_kl[k] / num_samples
    for k in range(n_layers):
        results[f"val_layerwise/r2a_lm_layer_{k}"] = r2a_lm[k] / num_samples
        results[f"val_layerwise/r2a_kl_layer_{k}"] = r2a_kl[k] / num_samples
    for k in range(n_layers + 1):
        results[f"val_layerwise/nmse_layer_{k}"] = nmse_by_layer[k] / num_samples

    return results


def save_checkpoint(model, tokenizer, output_dir):
    """Save full model checkpoint (no conversion needed)."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Checkpoint saved to {output_dir}")


def save_latest_checkpoint(model, tokenizer, base_dir, step):
    """Save checkpoint to base_dir/latest_step_N, removing any previous latest_step_* dir."""
    import glob
    import shutil
    # Remove previous latest checkpoint
    for old_dir in glob.glob(os.path.join(base_dir, "latest_step_*")):
        shutil.rmtree(old_dir)
    output_dir = os.path.join(base_dir, f"latest_step_{step}")
    save_checkpoint(model, tokenizer, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Train with bridging loss")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--learning_rate", "-lr", type=float, help="Override learning rate")
    parser.add_argument("--l1_weight", type=float, help="Override L1 weight")
    parser.add_argument("--debug_mode", nargs="?", const="true", default=None, help="Override debug_mode (--debug_mode, --debug_mode=true, --debug_mode=false). If activating debug mode through this setting, wandb will be disabled.")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Validate exactly one training mode is set
    has_bridging = config.bridging is not None
    has_direct = config.direct is not None
    if has_bridging == has_direct:
        raise ValueError("Config must specify exactly one of 'bridging' or 'direct' section.")

    # Apply overrides
    config_changed = False

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
        print(f"Override learning rate: {args.learning_rate}")
        config_changed = True

    if args.l1_weight is not None and config.transcoder:
        config.transcoder.l1_weight = args.l1_weight
        print(f"Override L1 weight: {args.l1_weight}")
        config_changed = True

    if args.debug_mode is not None:
        if args.debug_mode.lower() == "true":
            config.debug_mode = True
            config.use_wandb = False
            print("Using debug mode through flag: wandb disabled")
            if config.run_name_prefix and not config.run_name_prefix.endswith("_debug"):
                config.run_name_prefix += "_debug"
                print(f"Using debug mode through flag: added _debug to run_name_prefix, now '{config.run_name_prefix}'")
        elif args.debug_mode.lower() == "false":
            config.debug_mode = False
        else:
            parser.error(f"Invalid value for --debug_mode: '{args.debug_mode}'. Must be 'true' or 'false'.")
        print(f"Override debug_mode: {config.debug_mode}")

    if config_changed:
        # Update run name and output dir to reflect the overrides
        config.wandb_run_name = None  # Force regeneration
        config.output_dir = None      # Force regeneration
        config = _finalize_config(config)  # Regenerate names with new params
        print(f"Updated run name: {config.wandb_run_name}")
        print(f"Updated output dir: {config.output_dir}")

    # Print mode-specific info
    if config.direct:
        print("Starting direct fine-tuning")
        print(f"  Base model: {config.model_name}")
        print(f"  Copied tokens: {config.direct.copied_tokens}")
    else:
        assert config.bridging is not None
        print("Starting bridging training")
        print(f"  Base model: {config.model_name}")
        print(f"  Reference model: {config.bridging.reference_model_path}")
        print(f"  Loss type: {config.bridging.loss_type}")
        print(f"  N cutoffs: {config.bridging.n_cutoffs}")

    # Setup models
    if config.direct:
        model, ref_model, tokenizer = setup_models_direct(config)
    else:
        model, ref_model, tokenizer = setup_models_bridging(config)

    train_dataset, train_dataloader, val_dataloader = setup_data(config, tokenizer)
    optimizer, scheduler, total_steps, warmup_steps = setup_training(config, model, train_dataset)

    # WandB
    if config.use_wandb:
        mode_prefix = "direct" if config.direct else "bridging"
        wandb.init(
            project=config.wandb_project,
            name=f"{mode_prefix}_{config.wandb_run_name}",
            config=config.__dict__
        )

    print("Training setup:")
    print(f"  - Train dataset size: {len(train_dataset)}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Warmup steps: {warmup_steps}")

    # Training loop
    current_step = 0
    total_samples_seen = 0

    for epoch in range(config.num_epochs):
        epoch_loss, current_step, total_samples_seen = train_epoch(
            model, ref_model, tokenizer, train_dataloader, optimizer, scheduler,
            config,
            epoch, current_step, total_steps, total_samples_seen,
            val_dataloader=val_dataloader,
        )

        # Save checkpoint at end of epoch (overwrites previous latest)
        if config.save_checkpoints:
            save_latest_checkpoint(model, tokenizer, config.output_dir, current_step)

    print("Training complete!")

    # Always save final checkpoint
    save_checkpoint(model, tokenizer, config.output_dir)


if __name__ == "__main__":
    main()
