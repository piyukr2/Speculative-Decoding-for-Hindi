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


class BottleneckExitHead(nn.Module):
    """Compact exit head: LayerNorm → Linear(hidden→256) → GELU → Linear(256→vocab). ~39M params."""

    def __init__(self, hidden_size: int, vocab_size: int, bottleneck_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, vocab_size, bias=False)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"BottleneckExitHead: {total_params / 1e6:.1f}M params "
              f"({hidden_size}→{bottleneck_dim}→{vocab_size})")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.norm.weight)  # match device AND dtype
        x = self.norm(hidden_states)
        x = self.down_proj(x)
        x = self.act(x)
        return self.up_proj(x)


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
    # True early-exit forward (used during inference drafting)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def partial_forward(
        self,
        input_ids: torch.Tensor,
        depth: int,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
    ):
        """Run only the first `depth` layers of the base model, then apply exit head.

        Returns (logits, new_past_key_values).
        Does NOT run layers depth..L-1.

        Args:
            input_ids: [B, T] token ids (or [B, 1] when using KV cache).
            depth: number of transformer layers to run (1-indexed, runs layers 0..depth-1).
            position_ids: [B, T] position indices. Auto-computed from cache length if None.
            past_key_values: tuple of (key, value) tensors from a previous partial_forward call.

        Returns:
            logits: [B, T, V] exit-head logits after layer `depth`.
            new_past_key_values: tuple of (key, value) for layers 0..depth-1.
        """
        layers = self._get_transformer_layers()
        embed_tokens = self._get_embed_tokens()

        # --- Embedding ---
        hidden_states = embed_tokens(input_ids)
        device = hidden_states.device
        dtype = hidden_states.dtype
        batch_size, seq_len = input_ids.shape

        # --- Position IDs ---
        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0:
                # past_key_values[layer_idx] = (key, value), key shape: [B, num_heads, S, head_dim]
                past_len = past_key_values[0][0].shape[2]
            else:
                past_len = 0
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # --- Causal attention mask ---
        # Try the model's own mask builder first; fall back to None (let layers handle it)
        causal_mask = None
        try:
            base_inner = getattr(self.base_model, "model", None)
            if base_inner is not None and hasattr(base_inner, "_update_causal_mask"):
                total_len = (past_key_values[0][0].shape[2] if past_key_values else 0) + seq_len
                causal_mask = base_inner._update_causal_mask(
                    None, hidden_states, torch.arange(total_len, device=device),
                    past_key_values=None,
                )
        except Exception:
            causal_mask = None

        # --- Run layers 0..depth-1 ---
        new_past_key_values = []
        for i in range(depth):
            layer = layers[i]
            past_kv = past_key_values[i] if past_key_values is not None and i < len(past_key_values) else None

            layer_kwargs = {
                "hidden_states": hidden_states.to(layer.self_attn.q_proj.weight.device),
                "position_ids": position_ids.to(layer.self_attn.q_proj.weight.device),
                "use_cache": True,
            }

            # Pass attention mask if we have one
            if causal_mask is not None:
                layer_kwargs["attention_mask"] = causal_mask.to(layer.self_attn.q_proj.weight.device)

            # Pass past KV cache for this layer
            if past_kv is not None:
                layer_kwargs["past_key_value"] = past_kv

            layer_out = layer(**layer_kwargs)

            # layer_out is (hidden_states, present_kv, ...) or (hidden_states, ...)
            hidden_states = layer_out[0]
            if len(layer_out) > 1 and layer_out[1] is not None:
                new_past_key_values.append(layer_out[1])
            else:
                new_past_key_values.append(None)

        # --- Apply exit head ---
        hidden_states = hidden_states.to(device)
        logits = self.exit_heads[str(depth)](hidden_states)  # [B, T, V]

        return logits, tuple(new_past_key_values)

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
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.exit_heads.load_state_dict(state)
        self.exit_heads.to(next(self.base_model.parameters()).device)


class ThompsonSamplingController:
    """Dynamically selects exit depth using Thompson Sampling (Beta-Bernoulli bandit)."""

    def __init__(self, exit_depths: List[int], total_layers: int = 28):
        self.depths = list(exit_depths)
        self.total_layers = total_layers
        self.alpha = {d: 1.0 for d in self.depths}  # Beta prior successes
        self.beta = {d: 1.0 for d in self.depths}   # Beta prior failures

    def select_depth(self) -> int:
        import random
        best_depth = self.depths[0]
        best_score = -1.0
        for d in self.depths:
            theta = random.betavariate(self.alpha[d], self.beta[d])
            score = theta * (self.total_layers / d)  # acceptance_rate × speedup_factor
            if score > best_score:
                best_score = score
                best_depth = d
        return best_depth

    def update(self, depth: int, num_accepted: int, num_drafted: int):
        self.alpha[depth] += num_accepted
        self.beta[depth] += (num_drafted - num_accepted)


class WeightedThompsonSamplingController:
    """Thompson Sampling with explicit correctness/time reward weighting.

    Like the original ThompsonSamplingController, depth selection is driven by
    sampling from Beta posteriors.  The key difference is how the sampled
    acceptance-rate estimate is combined with a speed bonus:

        score = w_correctness * theta + w_time * (1 - depth / total_layers)

    Defaults: correctness 92%, time 8%  (within the 90-95 / 5-10 spec).
    The Beta prior is updated identically to the original controller
    (alpha += accepted, beta += rejected) so the underlying Thompson
    Sampling mechanism is unchanged.
    """

    def __init__(
        self,
        exit_depths: List[int],
        total_layers: int = 28,
        w_correctness: float = 0.92,
        w_time: float = 0.08,
    ):
        self.depths = list(exit_depths)
        self.total_layers = total_layers
        self.w_correctness = w_correctness
        self.w_time = w_time
        self.alpha = {d: 1.0 for d in self.depths}  # Beta prior successes
        self.beta = {d: 1.0 for d in self.depths}   # Beta prior failures

    def select_depth(self) -> int:
        import random
        best_depth = self.depths[0]
        best_score = -1.0
        for d in self.depths:
            theta = random.betavariate(self.alpha[d], self.beta[d])
            # Weighted score: correctness dominates, speed is a tiebreaker
            score = (self.w_correctness * theta
                     + self.w_time * (1.0 - d / self.total_layers))
            if score > best_score:
                best_score = score
                best_depth = d
        return best_depth

    def update(self, depth: int, num_accepted: int, num_drafted: int):
        self.alpha[depth] += num_accepted
        self.beta[depth] += (num_drafted - num_accepted)


class UCBController:
    """Dynamically selects exit depth using Upper Confidence Bound (UCB1).

    Reward definition:
      reward = w_correctness * (accepted/drafted) + w_time * (1 - depth/total_layers)

    Correctness gets 90-95% weight (default 0.92) and time/speed gets
    5-10% weight (default 0.08), so the algorithm strongly prioritises
    accuracy while still nudging towards faster (shallower) depths.
    """

    def __init__(
        self,
        exit_depths: List[int],
        total_layers: int = 28,
        w_correctness: float = 0.92,
        w_time: float = 0.08,
        c: float = 1.41,
    ):
        self.depths = list(exit_depths)
        self.total_layers = total_layers
        self.w_correctness = w_correctness
        self.w_time = w_time
        self.c = c  # exploration parameter (sqrt(2) ≈ 1.41 is standard UCB1)
        self.counts: Dict[int, int] = {d: 0 for d in self.depths}
        self.total_reward: Dict[int, float] = {d: 0.0 for d in self.depths}
        self.total_rounds: int = 0

    def select_depth(self) -> int:
        import math
        # Ensure each arm is tried at least once (initialisation phase)
        for d in self.depths:
            if self.counts[d] == 0:
                return d

        best_depth = self.depths[0]
        best_ucb = -float("inf")
        for d in self.depths:
            avg_reward = self.total_reward[d] / self.counts[d]
            exploration = self.c * math.sqrt(math.log(self.total_rounds) / self.counts[d])
            ucb_score = avg_reward + exploration
            if ucb_score > best_ucb:
                best_ucb = ucb_score
                best_depth = d
        return best_depth

    def update(self, depth: int, num_accepted: int, num_drafted: int):
        correctness = num_accepted / num_drafted if num_drafted > 0 else 0.0
        time_bonus = 1.0 - (depth / self.total_layers)
        reward = self.w_correctness * correctness + self.w_time * time_bonus
        self.counts[depth] += 1
        self.total_reward[depth] += reward
        self.total_rounds += 1
