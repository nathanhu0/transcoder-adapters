"""Configuration management for sparse adaptation experiments."""

import yaml
from dataclasses import dataclass, field
from typing import Any

from .dataset.openthoughts.config import OpenThoughtsConfig
from .dataset.gemma.config import FineWebLMSysMixedConfig

from .dataset.datasetspecific_config import DatasetSpecificConfig, DatasetType

from .dataset.PredefinedDataset import LengthExcessionBehavior
from pathlib import Path


@dataclass
class TranscoderConfig:
    """Transcoder adapter configuration."""
    n_features: int = 8192
    dec_bias: bool = True  # Whether to include bias in decoder
    l1_weight: float | None = 0.001  # Weight for L1 regularization on features
    normalize_by_layer: bool = False  # Whether to normalize L1 weights by layer output norm
    schedule_l1_weight: bool = False  # Whether to linearly ramp L1 weight from 0 to target weight
    pre_activation_loss_weight: float = 0.0  # Weight for pre-activation loss (prevents dead features)


@dataclass
class BridgingConfig:
    """Bridging loss configuration for distillation with layer-wise compatibility."""
    reference_model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    loss_type: str = "kl"  # "kl" (distillation to ref) or "lm" (language modeling)
    n_cutoffs: int = 1  # Number of layer cutoffs to sample per batch
    sampling: str = "uniform"  # "uniform" or list of fixed layer indices
    always_include_adapt_only: bool = True  # Always include clean adapter forward pass
    lambda_adapt: float = 1.0  # Weight on adapt-only loss (end-to-end KL/LM)
    lambda_bridge: float = 2.0  # Weight on bridging losses (mixed forward passes)
    lambda_nmse: float = 1.0  # Weight on activation matching loss (optional)
    backbone: str = "target"  # "base" or "target" (which model provides attn/embed/layernorm)


@dataclass
class DirectConfig:
    """Direct fine-tuning configuration (no bridging, LM loss only on response tokens)."""
    reference_model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Source for copied token embeddings
    copied_tokens: list[str] = field(default_factory=lambda: ["<think>", "</think>"])  # Tokens to add and copy from reference


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-Math-7B"
    model_arch: str | None = None  # "qwen2", "gemma2", etc. Auto-detected from model_name if None

    # Transcoder configuration
    transcoder: TranscoderConfig | None = None

    # Training mode (exactly one of bridging or direct must be set)
    bridging: BridgingConfig | None = None
    direct: DirectConfig | None = None

    # Training hyperparameters
    learning_rate: float = 8e-4
    batch_size: int = 1
    micro_batch_size: int | None = None # not really doing gradient accumulation anymore, see note in PredefinedDataset's _make_dataloader function. If None, this will be set to batch_size.
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    seed: int = 42

    # Data settings
    dataset_type: DatasetType = DatasetType.OPEN_THOUGHTS
    length_excession_behavior: LengthExcessionBehavior = LengthExcessionBehavior.TRUNCATE
    loss_on_prompt: bool = True
    dataset: DatasetSpecificConfig = OpenThoughtsConfig(
        data_path="/nlp/scr/nathu/sparse-adaptation/data/openthoughts/stratified_n55000_t10000_s42_train.jsonl",
        data_format="deepseek",
        max_seq_length=10000,
        val_data_path="/nlp/scr/nathu/sparse-adaptation/data/openthoughts/stratified_n55000_t10000_s42_val.jsonl"
    )

    val_frequency: int = 1000  # Run validation every N steps
    layerwise_val_frequency: int = 2000  # Run layerwise validation every N steps

    # Output settings (auto-computed)
    output_dir: str | None = None  # Will be computed from hyperparameters
    wandb_run_name: str | None = None  # Will be computed from hyperparams
    run_name_prefix: str | None = None  # Optional prefix for run name (e.g., "r1_distil_yolo") for use in wandb and the output dir.

    # WandB settings
    use_wandb: bool = True
    wandb_project: str = "sparse-adaptation"

    # Checkpoint settings
    save_checkpoints: bool = False  # If True, save periodic checkpoints (overwrites single 'latest' dir)
    checkpoint_frequency: int = 10000  # Save checkpoint every N steps

    # Debug settings
    debug_mode: bool = False  # If True, break after 50 steps for quick testing


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path) # type: ignore

    if not config_path.exists(): # type: ignore
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Handle nested configs
    adapter_configs = {}
    if 'transcoder' in config_dict:
        adapter_configs['transcoder'] = TranscoderConfig(**config_dict.pop('transcoder'))
    if 'bridging' in config_dict:
        adapter_configs['bridging'] = BridgingConfig(**config_dict.pop('bridging'))
    if 'direct' in config_dict:
        adapter_configs['direct'] = DirectConfig(**config_dict.pop('direct'))

    # Create main config with adapter configs
    config = ExperimentConfig(**config_dict, **adapter_configs)

    # Ensure numeric types are correct (YAML can load as strings)
    config.learning_rate = float(config.learning_rate)
    config.batch_size = int(config.batch_size)
    if config.micro_batch_size is None:
        config.micro_batch_size = config.batch_size # int(config.micro_batch_size) # We're not doing gradient accumulation, see note in PredefinedDataset's _make_dataloader function.

    if config.transcoder:
        # Convert transcoder weights to float if they exist
        if config.transcoder.l1_weight is not None:
            config.transcoder.l1_weight = float(config.transcoder.l1_weight)
        if config.transcoder.pre_activation_loss_weight is not None:
            config.transcoder.pre_activation_loss_weight = float(config.transcoder.pre_activation_loss_weight)

    # Resolve model_arch from model_name if not set
    if config.model_arch is None:
        from models import detect_architecture
        config.model_arch = detect_architecture(config.model_name)

    # Convert dataset_type from string to enum
    if isinstance(config.dataset_type, str):
        config.dataset_type = DatasetType(config.dataset_type)
    
    if isinstance(config.length_excession_behavior, str):
        config.length_excession_behavior = LengthExcessionBehavior(config.length_excession_behavior)

    # Parse dataset sub-dict
    if isinstance(config.dataset, dict):
        match config.dataset_type:
            case DatasetType.OPEN_THOUGHTS:
                config.dataset = OpenThoughtsConfig(**config.dataset)
            case DatasetType.FINEWEB_LMYSYSCHAT_MIXED:
                config.dataset = FineWebLMSysMixedConfig(**config.dataset)
            case _:
                raise ValueError(f"Unsupported dataset type: {config.dataset_type}")
    
    assert config.dataset.dataset_type == config.dataset_type, (
        f"Dataset type mismatch: you set dataset_type to {config.dataset_type}, but the provided dataset-specific config is for {config.dataset.dataset_type}. Please make sure these match."
    )

    # Print a warning if there were any extra keys in the YAML that were not used in the config dataclass
    extra_keys = set(config_dict.keys()) - set(ExperimentConfig.__dataclass_fields__.keys())
    if extra_keys:
        print("Warning: the following keys in the config file were not recognized and will be ignored:", extra_keys)

    # Auto-compute run name and output dir if not specified
    config = _finalize_config(config)
    return config


def _finalize_config(config: ExperimentConfig) -> ExperimentConfig:
    """Finalize config by computing run names and output directories."""

    # Build run name from hyperparameters
    if config.wandb_run_name is None:
        run_parts: list[str] = []

        # Use prefix if provided, otherwise generate default prefix
        if config.run_name_prefix:
            run_parts.append(config.run_name_prefix)
        else:
            model_size = _extract_model_size(config.model_name)
            run_parts.extend(["transcoder", model_size])

        # Add transcoder params
        if config.transcoder:
            run_parts.append(f"tc{config.transcoder.n_features}")
            if getattr(config.transcoder, 'dec_bias', False):
                run_parts.append("decb")
            if config.transcoder.l1_weight:
                run_parts.append(f"l1w{config.transcoder.l1_weight}")
            if getattr(config.transcoder, 'normalize_by_layer', False):
                run_parts.append("norm")
            if getattr(config.transcoder, 'schedule_l1_weight', False):
                run_parts.append("sch")
            if getattr(config.transcoder, 'pre_activation_loss_weight', 0.0) > 0:
                run_parts.append(f"pre{config.transcoder.pre_activation_loss_weight}")

        # Add mode-specific params
        if config.bridging:
            backbone = getattr(config.bridging, 'backbone', 'base')
            run_parts.append(f"{backbone[:3]}bb")  # "basbb" or "tgtbb"
            run_parts.append(f"lb{config.bridging.lambda_bridge}")
            run_parts.append(f"ln{config.bridging.lambda_nmse}")
        elif config.direct:
            run_parts.append("direct")

        # Add training params
        run_parts.append(f"lr{config.learning_rate:.0e}")
        run_parts.append(f"bs{config.batch_size}")

        config.wandb_run_name = "_".join(run_parts)

    # Build output directory
    if config.output_dir is None:
        import os
        from datetime import datetime
        user = os.environ.get("USER")
        if not user:
            raise RuntimeError("$USER environment variable is not set. Provide an output_dir in your config or set the USER environment variable so we know where to save checkpoints.")
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        config.output_dir = f"/nlp/scr/{user}/sparse-adaptation/checkpoints/{config.wandb_run_name}_{date_str}_{slurm_job_id}"
        print(f"Checkpoints save directory: {config.output_dir}")

    return config


def _extract_model_size(model_name: str) -> str:
    """Extract model size from model name (e.g., '7B' from 'Qwen/Qwen2.5-7B-Instruct')."""
    import re
    # Look for patterns like 1.5B, 3B, 7B, 14B, etc.
    match = re.search(r'(\d+(?:\.\d+)?[BMK])', model_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "unknown"


def apply_overrides(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply command line overrides to config.

    Supports:
    - Direct params: learning_rate=1e-3
    - Nested params: transcoder.n_features=2048, bridging.lambda_bridge=2.0
    """
    for key, value in overrides.items():
        if '.' in key:
            section, param = key.split('.', 1)
            if section == 'transcoder' and config.transcoder:
                setattr(config.transcoder, param, value)
            elif section == 'bridging' and config.bridging:
                setattr(config.bridging, param, value)
            elif section == 'direct' and config.direct:
                setattr(config.direct, param, value)
            else:
                raise ValueError(f"Invalid override section: {section}")
        else:
            # Handle top-level params
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")

    return config


def save_config(config: ExperimentConfig, output_path: str):
    """Save configuration to YAML file."""
    config_dict = {}
    for field_name in config.__dataclass_fields__:
        val = getattr(config, field_name)
        if hasattr(val, '__dataclass_fields__'):
            config_dict[field_name] = val.__dict__
        else:
            config_dict[field_name] = val

    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, indent=2, default_flow_style=False)
