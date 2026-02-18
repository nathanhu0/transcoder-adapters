"""
Wrapper around Qwen2ForCausalLMWithTranscoderRelP for attribution.

Provides the interface expected by the attribution loop (cfg, tokenizer,
setup_attribution, forward, unembed.W_U) while using the RelP model
for linearized backward passes.
"""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoTokenizer


@dataclass
class RelPModelConfig:
    """Config object matching what attribution code expects."""
    n_layers: int
    d_model: int
    d_vocab: int
    n_features: int  # transcoder features per layer
    device: torch.device
    dtype: torch.dtype
    tokenizer_name: str


class UnembedWrapper:
    """Wrapper to provide W_U interface expected by attribution code."""

    def __init__(self, lm_head_weight: torch.Tensor):
        # lm_head.weight: (vocab_size, d_model)
        # W_U should be (d_model, vocab_size)
        self.W_U = lm_head_weight.T


class RelPReplacementModel(nn.Module):
    """
    Wrapper around Qwen2ForCausalLMWithTranscoderRelP for attribution.

    This provides the minimal interface needed by the attribution loop:
    - cfg, tokenizer, scan
    - ensure_tokenized()
    - setup_attribution() -> RelPAttributionContext
    - forward() with hooks support
    - unembed.W_U for logit vectors

    The key difference from other wrappers: we use the RelP model's
    built-in linearization (via detach in forward) rather than
    computing RelP backward separately.
    """

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = self._make_config()
        # HF config exposed for upstream Graph constructor
        # (convert_nnsight_config_to_transformerlens needs .to_dict(), .architectures, etc.)
        self.hf_config = model.config
        self.scan = None  # No public feature database for custom transcoders

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: torch.device | str | None = None,
        device_map: str | dict | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "RelPReplacementModel":
        """Load a RelPReplacementModel from a local path or HF repo ID.

        Args:
            checkpoint_path: Local path or HF repo ID
            device: Device to load to (ignored if device_map is set)
            device_map: Device map for multi-GPU. Use "auto" to split layers across GPUs.
            dtype: Model dtype (default: bfloat16)
        """
        from models.qwen2_with_transcoder_relp import Qwen2ForCausalLMWithTranscoderRelP

        # Load model to CPU first, then dispatch
        model = Qwen2ForCausalLMWithTranscoderRelP.from_pretrained(checkpoint_path)
        model = model.to(dtype=dtype)

        if device_map is not None:
            # Multi-GPU: use accelerate to split layers across devices
            from accelerate import infer_auto_device_map, dispatch_model

            if device_map == "auto":
                device_map = infer_auto_device_map(
                    model,
                    max_memory={i: "70GiB" for i in range(torch.cuda.device_count())},
                    no_split_module_classes=["Qwen2DecoderLayerRelP"],
                )
                print(f"Auto device_map: {device_map}")

            model = dispatch_model(model, device_map=device_map)
        else:
            # Single GPU
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(device, str):
                device = torch.device(device)
            model = model.to(device=device)

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        return cls(model, tokenizer)

    def _make_config(self) -> RelPModelConfig:
        """Build config from model."""
        hf_config = self.model.config

        # Get n_features from first MLP layer
        n_features = self.model.model.layers[0].mlp.n_features

        return RelPModelConfig(
            n_layers=hf_config.num_hidden_layers,
            d_model=hf_config.hidden_size,
            d_vocab=hf_config.vocab_size,
            n_features=n_features,
            device=next(self.model.parameters()).device,
            dtype=next(self.model.parameters()).dtype,
            tokenizer_name=getattr(hf_config, '_name_or_path', 'unknown'),
        )

    @property
    def unembed(self) -> UnembedWrapper:
        """Unembedding matrix wrapper."""
        return UnembedWrapper(self.model.lm_head.weight)

    @property
    def W_E(self) -> torch.Tensor:
        """Embedding matrix (vocab_size, d_model)."""
        return self.model.model.embed_tokens.weight

    def ln_final(self, x: torch.Tensor) -> torch.Tensor:
        """Apply final layer norm."""
        return self.model.model.norm(x)

    def ensure_tokenized(self, prompt: str | torch.Tensor | list[int]) -> torch.Tensor:
        """Convert prompt to token tensor with BOS handling."""
        if isinstance(prompt, str):
            tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.squeeze()
        elif isinstance(prompt, list):
            tokens = torch.tensor(prompt, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        if tokens.ndim > 1:
            raise ValueError(f"Tokens must be 1D, got {tokens.shape}")

        # Add BOS if not present
        if tokens[0] not in self.tokenizer.all_special_ids:
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            if bos is not None:
                tokens = torch.cat([torch.tensor([bos]), tokens])

        return tokens.to(self.cfg.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        """Forward pass, returns logits."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        outputs = self.model(input_ids)
        return outputs.logits

    def get_encoder_vecs(self) -> torch.Tensor:
        """Get encoder vectors for all layers, shape (n_layers, n_features, d_model)."""
        encoder_vecs = []
        for layer in self.model.model.layers:
            # transcoder_enc.weight: (n_features, d_model)
            encoder_vecs.append(layer.mlp.transcoder_enc.weight.detach())
        return torch.stack(encoder_vecs)

    def get_decoder_vecs(self) -> torch.Tensor:
        """Get decoder vectors for all layers, shape (n_layers, n_features, d_model)."""
        decoder_vecs = []
        for layer in self.model.model.layers:
            # transcoder_dec.weight: (d_model, n_features) -> transpose
            decoder_vecs.append(layer.mlp.transcoder_dec.weight.T.detach())
        return torch.stack(decoder_vecs)

    @torch.no_grad()
    def setup_attribution(self, prompt: str | torch.Tensor):
        """
        Run forward pass and prepare for attribution.

        Returns RelPAttributionContext with:
        - activation_matrix: sparse tensor of active features
        - encoder_vecs: for gradient injection
        - token_vectors: embedding vectors
        - logits: model output
        """
        # Import here to avoid circular dependency
        from analysis.attribution.relp_context import RelPAttributionContext

        tokens = self.ensure_tokenized(prompt)

        # Forward pass - this caches features in each layer's MLP
        logits = self.forward(tokens)

        # Build activation matrix from cached features
        cached = self.model.get_cached_features()
        n_layers = self.cfg.n_layers
        n_pos = len(tokens)
        n_features = self.cfg.n_features

        # Find active (non-zero) features
        indices_list = []
        values_list = []

        for layer_idx, features in cached.items():
            # features: (1, n_pos, n_features) -> squeeze batch
            feats = features.squeeze(0)  # (n_pos, n_features)

            # Find non-zero entries
            pos_idx, feat_idx = torch.where(feats > 0)

            for p, f in zip(pos_idx, feat_idx):
                indices_list.append([layer_idx, p.item(), f.item()])
                values_list.append(feats[p, f].item())

        if indices_list:
            indices = torch.tensor(indices_list, device=self.cfg.device).T  # (3, nnz)
            values = torch.tensor(values_list, device=self.cfg.device)
        else:
            indices = torch.zeros((3, 0), dtype=torch.long, device=self.cfg.device)
            values = torch.zeros(0, device=self.cfg.device)

        activation_matrix = torch.sparse_coo_tensor(
            indices, values,
            size=(n_layers, n_pos, n_features),
            device=self.cfg.device
        ).coalesce()

        # Get encoder vecs for active features (for gradient injection)
        all_encoder_vecs = self.get_encoder_vecs()  # (n_layers, n_features, d_model)

        if activation_matrix._nnz() > 0:
            feat_layers, feat_pos, feat_idx = activation_matrix.indices()
            encoder_vecs = all_encoder_vecs[feat_layers, feat_idx]  # (nnz, d_model)
        else:
            encoder_vecs = torch.zeros((0, self.cfg.d_model), device=self.cfg.device)

        # Token embeddings
        token_vectors = self.W_E[tokens].detach()

        return RelPAttributionContext(
            model=self,
            tokens=tokens,
            activation_matrix=activation_matrix,
            encoder_vecs=encoder_vecs,
            token_vectors=token_vectors,
            logits=logits,
        )
