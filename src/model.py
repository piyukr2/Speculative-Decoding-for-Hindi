"""
Early-Exit Speculative Decoding (EESD) model wrapper.

Adds lightweight exit heads at specified transformer layers.
During training, only the exit heads are updated; the base model is frozen.
During inference, the shallowest exit head acts as the draft model.

Reference: Liu et al. (2024a) "Speculative Decoding via Early-Exiting for
Faster LLM Inference with Thompson Sampling Control Mechanism" (ACL Findings 2024)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from typing import List, Dict, Optional


class ExitHead(nn.Module):
    """Lightweight exit head: LayerNorm → Linear(hidden → vocab)."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.norm.weight)  # match device AND dtype
        return self.lm_head(self.norm(hidden_states))


class EarlyExitLM(nn.Module):
    """
    Wraps a causal LM with early-exit heads at specified layer depths.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        exit_depths: Transformer layer indices where exit heads are inserted.
                     E.g. [2, 4, 6] means heads after layers 2, 4, and 6.
        draft_depth: Which exit depth to use during speculative drafting.
    """

    def __init__(
        self,
        model_name_or_path: str,
        exit_depths: List[int] = [8, 16, 22],
        draft_depth: Optional[int] = None,
        load_in_8bit: bool = False,
    ):
        super().__init__()
        self.exit_depths = sorted(exit_depths)
        self.draft_depth = draft_depth if draft_depth is not None else self.exit_depths[0]
        assert self.draft_depth in self.exit_depths, (
            f"draft_depth {self.draft_depth} must be in exit_depths {self.exit_depths}"
        )

        # Load base model (frozen during exit-head training)
        load_kwargs = {
            "device_map": "auto",
        }
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        else:
            # Newer transformers (>=4.45) renamed torch_dtype → dtype
            import transformers
            _ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
            if _ver >= (4, 45):
                load_kwargs["dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float16
            
        self.base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        hidden_size = self.base_model.config.hidden_size
        vocab_size = self.base_model.config.vocab_size

        # One exit head per depth
        self.exit_heads = nn.ModuleDict(
            {str(d): ExitHead(hidden_size, vocab_size) for d in self.exit_depths}
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_transformer_layers(self) -> nn.ModuleList:
        """Return the list of transformer decoder layers from the base model."""
        model = self.base_model
        # Works for Qwen2, LLaMA, Mistral, GPT-NeoX style models
        for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = model
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise RuntimeError(
            "Cannot locate transformer layers. "
            "Override _get_transformer_layers() for your model architecture."
        )

    def _get_embed_tokens(self) -> nn.Embedding:
        model = self.base_model
        for attr in ("model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"):
            obj = model
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise RuntimeError("Cannot locate embedding layer.")

    # ------------------------------------------------------------------
    # Forward pass (used during training)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_exit_logits: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the full forward pass and collect exit-head logits.

        Uses forward hooks to capture intermediate hidden states at each exit
        depth during a single base-model forward pass.  This avoids manual
        layer-by-layer iteration (which would break attention-mask preparation
        and rotary embeddings inside modern transformer layers).

        Returns:
            dict with keys:
              - "full_logits": final LM head logits [B, T, V]
              - "exit_logits": dict mapping str(depth) -> logits [B, T, V]
        """
        layers = self._get_transformer_layers()
        captured: Dict[str, torch.Tensor] = {}
        hooks = []

        if return_exit_logits:
            def make_hook(depth: int):
                def hook(module, input, output):
                    # output is (hidden_states, ...) for most transformer layers
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Keep native dtype (float16) to save VRAM on cloud GPUs
                    captured[str(depth)] = hidden
                return hook

            for depth in self.exit_depths:
                h = layers[depth - 1].register_forward_hook(make_hook(depth))
                hooks.append(h)

        # Single base-model forward pass (frozen, no grad needed for base weights)
        with torch.no_grad():
            full_out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()

        # Apply exit heads outside no_grad so gradients flow to head parameters
        exit_logits: Dict[str, torch.Tensor] = {}
        for depth in self.exit_depths:
            if str(depth) in captured:
                exit_logits[str(depth)] = self.exit_heads[str(depth)](captured[str(depth)])

        return {"full_logits": full_out.logits, "exit_logits": exit_logits}

    # ------------------------------------------------------------------
    # Draft generation (used during inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def draft(
        self,
        input_ids: torch.Tensor,
        K: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate K draft tokens using the exit head at self.draft_depth.

        Uses a forward hook on the base model to capture the hidden state at
        `draft_depth` without manually iterating layers (which would break
        attention-mask preparation inside modern attention layers).

        Note: the base model still runs all layers; the early-exit speedup at
        full scale requires a custom CUDA kernel or model surgery.  For this
        research prototype the hook approach gives correct token statistics
        (acceptance rate, morphological analysis) without kernel modifications.

        Args:
            input_ids: [1, T] prompt token ids.
            K: number of draft tokens to generate.
            temperature: sampling temperature (0.0 = greedy argmax).

        Returns:
            draft_ids: [1, K] draft token ids.
        """
        layers = self._get_transformer_layers()
        exit_head = self.exit_heads[str(self.draft_depth)]

        draft_ids = []
        cur_ids = input_ids.clone()

        for _ in range(K):
            captured_hidden: Dict[str, torch.Tensor] = {}

            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured_hidden["h"] = h.float()

            handle = layers[self.draft_depth - 1].register_forward_hook(hook)
            self.base_model(input_ids=cur_ids)
            handle.remove()

            # Use last-token hidden state for next-token prediction
            last_hidden = captured_hidden["h"][:, -1:, :]  # [1, 1, D] float32
            logits = exit_head(last_hidden)  # [1, 1, V]

            if temperature <= 0.0:
                next_token = logits.argmax(dim=-1)  # [1, 1]
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)

            draft_ids.append(next_token)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)

        return torch.cat(draft_ids, dim=1)  # [1, K]

    @torch.no_grad()
    def verify(
        self,
        input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Verify draft tokens with the full model in a single forward pass.

        Returns:
            verified_ids: accepted token ids (up to and including first mismatch).
        """
        full_input = torch.cat([input_ids, draft_ids], dim=1)  # [1, T+K]
        out = self.base_model(input_ids=full_input)
        full_logits = out.logits  # [1, T+K, V]

        # The full-model prediction for position T+i corresponds to logits at T+i-1
        T = input_ids.size(1)
        K = draft_ids.size(1)

        accepted = []
        for i in range(K):
            full_token = full_logits[:, T + i - 1, :].argmax(dim=-1)  # [1]
            draft_token = draft_ids[:, i]  # [1]
            if full_token.item() == draft_token.item():
                accepted.append(draft_token)
            else:
                accepted.append(full_token)  # accept the correct token
                break

        return torch.stack(accepted, dim=1) if accepted else draft_ids[:, :0]

    def trainable_parameters(self):
        """Only the exit heads are trainable."""
        return self.exit_heads.parameters()

    def save_exit_heads(self, path: str):
        torch.save(self.exit_heads.state_dict(), path)

    def load_exit_heads(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.exit_heads.load_state_dict(state)
        self.exit_heads.to(next(self.base_model.parameters()).device)
