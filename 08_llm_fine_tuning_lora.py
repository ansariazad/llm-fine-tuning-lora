#!/usr/bin/env python3
"""
08_llm_fine_tuning_lora.py — LoRA Fine-Tuning from Scratch in Pure PyTorch
===========================================================================
Reference Sources:
  - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
    https://arxiv.org/abs/2106.09685
  - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
    https://arxiv.org/abs/2305.14314
  - Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (2019)
    https://arxiv.org/abs/1902.00751

Implemented Techniques:
  1. LoRA Layer         — low-rank decomposition: W + ΔW = W + B×A (rank r ≪ d)
  2. LoRA Linear        — drop-in nn.Linear replacement with LoRA bypass
  3. LoRA Model Wrapper — inject LoRA into any nn.Module's target layers
  4. Parameter Stats    — count trainable vs frozen parameters
  5. Training Loop      — standard AdamW loop, only LoRA params updated
  6. Adapter Fusion     — merge LoRA weights back into base model (W += αBA)
  7. Rank ablation      — benchmark quality vs rank r ∈ {1, 2, 4, 8, 16}
  8. QLoRA concepts     — NF4 quantization explained with NumPy demo

Usage:
  python3 08_llm_fine_tuning_lora.py          # offline mode
  pip install torch && python3 08_...py       # full PyTorch mode
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. LoRA Core Layer
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class LoRALayer(nn.Module):
        """
        LoRA bypass: ΔW = B × A, where B ∈ R^{d_out × r}, A ∈ R^{r × d_in}
        and r ≪ min(d_in, d_out).

        Forward: output = W·x + (α/r) · B·A·x
          - W is frozen (requires_grad=False)
          - Only A and B are trained
          - Scaling by α/r keeps output magnitude stable across ranks
        """

        def __init__(self, d_in: int, d_out: int, rank: int = 4, alpha: float = 1.0):
            super().__init__()
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank

            # LoRA matrices — small, trainable
            self.lora_A = nn.Parameter(torch.randn(rank, d_in) * (1.0 / math.sqrt(d_in)))
            self.lora_B = nn.Parameter(torch.zeros(d_out, rank))  # B initialized to 0 → ΔW starts at 0

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # (batch, seq, d_in) @ A.T → (batch, seq, rank)
            # (batch, seq, rank) @ B.T → (batch, seq, d_out)
            return self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

    class LoRALinear(nn.Module):
        """
        Drop-in replacement for nn.Linear that adds a LoRA bypass.

        output = W·x + (α/r)·B·A·x
        Only lora_A and lora_B require gradients.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 4,
            alpha: float = 1.0,
            bias: bool = True,
        ):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.lora = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)

            # Freeze base weights
            self.linear.weight.requires_grad_(False)
            if self.linear.bias is not None:
                self.linear.bias.requires_grad_(False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.linear(x) + self.lora(x)

        def merge_weights(self) -> "nn.Linear":
            """
            Fusion: W_merged = W + (α/r) · B · A
            After merging, LoRA overhead is gone — same speed as base model.
            """
            with torch.no_grad():
                delta_W = self.lora.scaling * (self.lora.lora_B @ self.lora.lora_A)
                merged = nn.Linear(self.linear.in_features, self.linear.out_features,
                                   bias=self.linear.bias is not None)
                merged.weight.copy_(self.linear.weight + delta_W)
                if self.linear.bias is not None:
                    merged.bias.copy_(self.linear.bias)
            return merged

    # ---------------------------------------------------------------------------
    # 2. LoRA Model Wrapper
    # ---------------------------------------------------------------------------

    def inject_lora(
        model: "nn.Module",
        target_modules: List[str],
        rank: int = 4,
        alpha: float = 1.0,
    ) -> "nn.Module":
        """
        Replaces named Linear layers with LoRALinear in-place.
        Freezes all other parameters.
        """
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad_(False)

        # Replace target modules
        for name, module in list(model.named_modules()):
            if any(target in name for target in target_modules):
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model if not parent_name else dict(model.named_modules())[parent_name]
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALinear(
                        module.in_features,
                        module.out_features,
                        rank=rank,
                        alpha=alpha,
                        bias=module.bias is not None,
                    )
                    lora_layer.linear.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        lora_layer.linear.bias.data.copy_(module.bias.data)
                    setattr(parent, child_name, lora_layer)

        return model

    # ---------------------------------------------------------------------------
    # 3. Demo Transformer for Injection
    # ---------------------------------------------------------------------------

    class TinyTransformer(nn.Module):
        """A minimal transformer to demonstrate LoRA injection."""

        def __init__(self, vocab_size: int = 256, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "attn_q": nn.Linear(d_model, d_model, bias=False),
                    "attn_k": nn.Linear(d_model, d_model, bias=False),
                    "attn_v": nn.Linear(d_model, d_model, bias=False),
                    "attn_o": nn.Linear(d_model, d_model, bias=False),
                    "ffn_1":  nn.Linear(d_model, d_model * 4, bias=False),
                    "ffn_2":  nn.Linear(d_model * 4, d_model, bias=False),
                    "norm1":  nn.LayerNorm(d_model),
                    "norm2":  nn.LayerNorm(d_model),
                })
                for _ in range(n_layers)
            ])
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.embed(x)
            for layer in self.layers:
                # Simplified attention (no masking for brevity)
                q = layer["attn_q"](layer["norm1"](x))
                k = layer["attn_k"](layer["norm1"](x))
                v = layer["attn_v"](layer["norm1"](x))
                attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.size(-1)), dim=-1)
                x = x + layer["attn_o"](attn @ v)
                x = x + layer["ffn_2"](F.relu(layer["ffn_1"](layer["norm2"](x))))
            return self.lm_head(x)

    # ---------------------------------------------------------------------------
    # 4. Parameter Counter
    # ---------------------------------------------------------------------------

    def count_parameters(model: "nn.Module") -> Dict[str, int]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_pct": round(trainable / total * 100, 2),
        }

    # ---------------------------------------------------------------------------
    # 5. LoRA Training Loop
    # ---------------------------------------------------------------------------

    def train_lora(
        model: "nn.Module",
        n_steps: int = 50,
        batch_size: int = 4,
        seq_len: int = 16,
        vocab_size: int = 256,
        lr: float = 1e-3,
    ) -> List[float]:
        """
        Standard training loop — only LoRA parameters receive gradient updates.
        """
        # Only optimize trainable (LoRA) params
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
        losses = []
        model.train()
        for step in range(n_steps):
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            y = torch.randint(0, vocab_size, (batch_size, seq_len))
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    # ---------------------------------------------------------------------------
    # 6. Rank Ablation Study
    # ---------------------------------------------------------------------------

    def rank_ablation(ranks: List[int] = [1, 2, 4, 8, 16]) -> None:
        print(f"\n{'Rank r':<10} {'Trainable params':>18} {'Trainable %':>14} {'Loss/step':>12}")
        print("─" * 56)
        for r in ranks:
            model = TinyTransformer(vocab_size=256, d_model=64, n_layers=2)
            inject_lora(model, target_modules=["attn_q", "attn_v"], rank=r, alpha=float(r))
            stats = count_parameters(model)
            losses = train_lora(model, n_steps=20, vocab_size=256)
            print(
                f"{r:<10} {stats['trainable']:>18,} {stats['trainable_pct']:>13.2f}% "
                f"{sum(losses)/len(losses):>11.4f}"
            )


# ---------------------------------------------------------------------------
# 7. QLoRA Concepts — NF4 Quantization Demo (NumPy)
# ---------------------------------------------------------------------------

def qlora_nf4_demo() -> None:
    """
    Demonstrates the core idea of NF4 (Normal Float 4-bit) quantization from QLoRA.
    NF4 is information-theoretically optimal for normally distributed weights.
    """
    if not NUMPY_AVAILABLE:
        print("[QLoRA] NumPy not available — skipping NF4 demo")
        return

    # NF4 quantization levels (from QLoRA paper Table 1)
    NF4_LEVELS = [
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230, 1.0,
    ]

    def nf4_quantize(weights: "np.ndarray") -> Tuple["np.ndarray", float, float]:
        """Quantize float32 weights to NF4."""
        absmax = np.abs(weights).max()
        if absmax == 0:
            return np.zeros_like(weights, dtype=np.uint8), 0.0, 0.0
        normalized = weights / absmax  # scale to [-1, 1]

        quantized = np.zeros(len(weights), dtype=np.uint8)
        for i, w in enumerate(normalized):
            nearest = min(range(16), key=lambda j: abs(NF4_LEVELS[j] - w))
            quantized[i] = nearest

        return quantized, absmax, 1 / absmax

    def nf4_dequantize(quantized: "np.ndarray", absmax: float) -> "np.ndarray":
        """Reconstruct float32 from NF4 codes."""
        return np.array([NF4_LEVELS[int(q)] for q in quantized]) * absmax

    # Demo
    np.random.seed(42)
    original = np.random.randn(16).astype(np.float32)
    quantized, absmax, _ = nf4_quantize(original)
    dequantized = nf4_dequantize(quantized, absmax)
    error = np.abs(original - dequantized).mean()

    orig_bytes = original.nbytes
    quant_bytes = math.ceil(len(quantized) * 4 / 8)  # 4 bits per value

    print(f"\n  NF4 Quantization Demo (16 weights):")
    print(f"  Original float32  : {orig_bytes} bytes")
    print(f"  NF4 (4-bit)       : {quant_bytes} bytes")
    print(f"  Compression ratio : {orig_bytes / quant_bytes:.1f}×")
    print(f"  Mean abs error    : {error:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'━'*60}")
    print("  LLM FINE-TUNING — LoRA FROM SCRATCH")
    print(f"{'━'*60}\n")

    if not TORCH_AVAILABLE:
        print("⚠  PyTorch not installed — running concepts-only demo\n")
        print("  LoRA core idea:")
        print("    W_fine-tuned = W_pretrained + ΔW")
        print("    ΔW = B × A   where B ∈ R^{d×r}, A ∈ R^{r×d}, r ≪ d")
        print("    Only A and B are trained — base model W is frozen\n")
        print("  Parameter savings example (d=4096, r=8):")
        d, r = 4096, 8
        full_params = d * d
        lora_params = d * r + r * d
        print(f"    Full fine-tune : {full_params:,} params")
        print(f"    LoRA (r={r})    : {lora_params:,} params")
        print(f"    Reduction      : {full_params/lora_params:.0f}×")
        qlora_nf4_demo()
        return

    # --- Full PyTorch demo ---
    print("[1] Building TinyTransformer (vocab=256, d_model=64, n_layers=2)")
    model = TinyTransformer(vocab_size=256, d_model=64, n_layers=2)
    base_stats = count_parameters(model)
    print(f"    Base params  : {base_stats['total']:,}")

    print("\n[2] Injecting LoRA (rank=4, target=attn_q + attn_v)")
    inject_lora(model, target_modules=["attn_q", "attn_v"], rank=4, alpha=4.0)
    lora_stats = count_parameters(model)
    print(f"    Total params    : {lora_stats['total']:,}")
    print(f"    Trainable (LoRA): {lora_stats['trainable']:,} ({lora_stats['trainable_pct']}%)")
    print(f"    Frozen (base)   : {lora_stats['frozen']:,}")

    print("\n[3] Training (50 steps, only LoRA params)")
    losses = train_lora(model, n_steps=50, vocab_size=256)
    print(f"    Loss step  0 : {losses[0]:.4f}")
    print(f"    Loss step 25 : {losses[24]:.4f}")
    print(f"    Loss step 50 : {losses[49]:.4f}")

    print("\n[4] Rank Ablation Study")
    rank_ablation(ranks=[1, 2, 4, 8, 16])

    print("\n[5] QLoRA — NF4 4-bit Quantization Demo")
    qlora_nf4_demo()

    print(f"\n{'━'*60}\n")


if __name__ == "__main__":
    main()
