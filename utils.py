"""
Implementation follows the training specification in: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"(https://arxiv.org/abs/2402.17764)
and the companion training guide: The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

Key design points from the paper
─────────────────────────────────
1. BitLinear replaces every nn.Linear inside attention and the SwiGLU FFN.
2. The weight quantisation uses absmean scaling then round-clamp to {-1, 0, +1}.
3. Activations are per-token absmax-scaled to 8-bit integers (range [-127, 127]).
4. The layer owns an internal RMSNorm over the inputs so the pre-existing
   RMSNorms in LlamaDecoderLayer (input_layernorm / post_attention_layernorm)
   MUST be removed after conversion.
5. High-precision (bf16/fp32) shadow weights are kept during training; only
   the ternary weights are needed at inference time.
6. The straight-through estimator (STE) lets gradients flow through the
   non-differentiable round() calls.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


# ──────────────────────────────────────────────────────────────────────────────
# Quantisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _quantize_weights_ternary(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Absmean weight quantisation → {-1, 0, +1}.

    scale = mean(|W|) + ε          (per weight-matrix, scalar)
    W_q   = round_clamp(W / scale)  via STE

    Returns (W_quantized_float, scale) where W_quantized_float still lives in
    fp32/bf16 so normal matmul works during training.
    """
    scale = w.abs().mean().clamp(min=1e-8)
    # STE: forward = round, backward = identity
    w_scaled = w / scale
    w_q = (w_scaled.round().clamp(-1, 1) - w_scaled).detach() + w_scaled
    return w_q, scale


def _quantize_activations_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token absmax activation quantisation to int8 range [-127, 127].

    scale = max(|x|, dim=-1) / 127  (per token, shape [batch, seq, 1])
    x_q   = round_clamp(x / scale)  via STE
    """
    # x shape: [..., hidden_dim]
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    x_scaled = x / scale
    x_q = (x_scaled.round().clamp(-127, 127) - x_scaled).detach() + x_scaled
    return x_q, scale


# ──────────────────────────────────────────────────────────────────────────────
# RMSNorm (parameter-free version used internally by BitLinear)
# ──────────────────────────────────────────────────────────────────────────────

class _RMSNorm(nn.Module):
    """
    Root-mean-square layer normalisation without a learnable scale/bias,
    as used inside BitLinear per the training manuscript.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms


# ──────────────────────────────────────────────────────────────────────────────
# BitLinear
# ──────────────────────────────────────────────────────────────────────────────

class BitLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear implementing BitNet b1.58 training.

    Forward pass (training):
      1. Normalise input with internal parameter-free RMSNorm.
      2. Quantise normalised input activations to 8-bit per token (STE).
      3. Quantise weight matrix to ternary {-1, 0, +1} via absmean (STE).
      4. Compute F.linear(x_q, w_q) using the regular fp matmul kernel
         (specialised ternary kernels only needed at inference time).
      5. Rescale output: multiply by (weight_scale × activation_scale).

    Bias is always disabled as per the paper.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,          # paper: no bias
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.norm = _RMSNorm(in_features)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Internal RMSNorm
        x_normed = self.norm(x)

        # 2. Quantise activations (STE, keeps gradients)
        x_q, act_scale = _quantize_activations_int8(x_normed)

        # 3. Quantise weights (STE, keeps gradients through round)
        w_q, w_scale = _quantize_weights_ternary(self.weight)

        # 4. Linear using the (still fp) quantised values
        out = F.linear(x_q, w_q, self.bias)

        # 5. Rescale: broadcast act_scale [..., 1] and scalar w_scale
        out = out * act_scale * w_scale

        return out

    # ------------------------------------------------------------------
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "BitLinear":
        """Construct a BitLinear from an existing nn.Linear, copying its weight."""
        bl = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            bl.weight.copy_(linear.weight)
            if linear.bias is not None and bl.bias is not None:
                bl.bias.copy_(linear.bias)
        return bl


# ──────────────────────────────────────────────────────────────────────────────
# Architecture conversion
# ──────────────────────────────────────────────────────────────────────────────

def _replace_linear_in_module(module: nn.Module) -> None:
    """
    Recursively replaces every nn.Linear inside *module* with BitLinear,
    skipping the token embedding and lm_head (which stay full-precision).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, BitLinear.from_linear(child))
        else:
            _replace_linear_in_module(child)


def _remove_decoder_prenorms(model: LlamaForCausalLM) -> None:
    """
    Remove the pre-attention and pre-FFN RMSNorms from every decoder layer.

    Per the paper: "Remove RMSNorm before attention and SwiGLU because
    BitLinear has built-in RMSNorm."

    The norms are replaced with nn.Identity so that residual stream
    shapes remain unchanged and no index errors occur in the forward pass.
    """
    for layer in model.model.layers:
        if isinstance(layer, LlamaDecoderLayer):
            layer.input_layernorm = nn.Identity()
            layer.post_attention_layernorm = nn.Identity()


def convert_to_bitnet(model: LlamaForCausalLM) -> LlamaForCausalLM:
    """
    Convert a HuggingFace LlamaForCausalLM to a BitNet b1.58 model in-place.

    Steps (exactly as described in the training manuscript):
      1. Replace all nn.Linear layers inside each decoder layer's
         self-attention and MLP with BitLinear.
         (embed_tokens and lm_head are intentionally left as standard Linear.)
      2. Remove the input_layernorm and post_attention_layernorm from every
         LlamaDecoderLayer (BitLinear owns the normalisation).

    Args:
        model: A LlamaForCausalLM instance (typically freshly initialised
               from a config, or loaded from a checkpoint for fine-tuning).

    Returns:
        The same model object, mutated in-place, for convenience.
    """
    for layer in model.model.layers:
        if not isinstance(layer, LlamaDecoderLayer):
            continue
        # Replace Linear → BitLinear in attention and MLP blocks
        _replace_linear_in_module(layer.self_attn)
        _replace_linear_in_module(layer.mlp)

    # Remove pre-attention / pre-FFN norms (BitLinear has its own)
    _remove_decoder_prenorms(model)

    return model