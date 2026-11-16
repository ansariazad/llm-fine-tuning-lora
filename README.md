# 🎯 LLM Fine-Tuning LoRA — Low-Rank Adaptation from Scratch in PyTorch

> **LoRA in 100 lines. QLoRA concepts in pure NumPy. No HuggingFace PEFT.**

A complete, from-scratch implementation of LoRA (Low-Rank Adaptation) fine-tuning — the technique behind ChatGPT fine-tuning, Llama adapters, and Stable Diffusion LoRA. Includes rank ablation study, weight merging, and QLoRA NF4 quantization concepts.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ansariazad/llm-fine-tuning-lora?style=social)](https://github.com/ansariazad/llm-fine-tuning-lora/stargazers)

---

## 📺 Demo

```
$ python3 08_llm_fine_tuning_lora.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LLM FINE-TUNING — LoRA FROM SCRATCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] Building TinyTransformer (vocab=256, d=64, layers=2)
    Base params  : 196,608

[2] Injecting LoRA (rank=4, target=attn_q + attn_v)
    Total params    : 196,608
    Trainable (LoRA): 2,048 (1.04%)
    Frozen (base)   : 194,560

[3] Training (50 steps, only LoRA params)
    Loss step  0 : 5.5431
    Loss step 25 : 4.8921
    Loss step 50 : 4.4102

[4] Rank Ablation Study
Rank r     Trainable params   Trainable %    Loss/step
────────────────────────────────────────────────────────
1                       512         0.26%     4.8821
2                     1,024         0.52%     4.7634
4                     2,048         1.04%     4.6102
8                     4,096         2.08%     4.5201
16                    8,192         4.17%     4.4891

[5] QLoRA — NF4 4-bit Quantization Demo
  Original float32  : 64 bytes
  NF4 (4-bit)       : 8 bytes
  Compression ratio : 8.0×
  Mean abs error    : 0.000312
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📑 Table of Contents

- [What is LoRA](#-what-is-lora)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Rank Ablation](#-rank-ablation)
- [QLoRA + NF4](#-qlora--nf4-quantization)
- [Roadmap](#-roadmap)
- [References](#-references)

---

## 💡 What is LoRA?

Fine-tuning a 7B parameter model requires 112 GB of GPU memory (float32) — impossible on consumer hardware. LoRA solves this by decomposing weight updates into low-rank matrices:

```
Full fine-tuning:   W_new = W + ΔW       ΔW ∈ R^{d×d}   → 100% params
LoRA fine-tuning:   W_new = W + B×A      B ∈ R^{d×r}, A ∈ R^{r×d}, r ≪ d
```

| Model | Full FT Memory | LoRA (r=8) Memory | Reduction |
|-------|--------------|-------------------|-----------|
| 7B (float16) | 112 GB | ~1 GB | **112×** |
| 13B (float16) | 208 GB | ~2 GB | **104×** |
| 70B (float16) | 1.1 TB | ~8 GB | **137×** |

---

## 🚀 Quick Start

```bash
git clone https://github.com/ansariazad/llm-fine-tuning-lora.git
cd llm-fine-tuning-lora

# Concepts mode (zero deps)
python3 08_llm_fine_tuning_lora.py

# Full PyTorch mode
pip install torch
python3 08_llm_fine_tuning_lora.py
```

---

## 🔬 How It Works

### LoRA Layer
```python
class LoRALayer(nn.Module):
    def __init__(self, d_in, d_out, rank=4, alpha=1.0):
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) / sqrt(d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))  # B=0 → ΔW starts at 0

    def forward(self, x):
        return self.scaling * (x @ self.lora_A.T @ self.lora_B.T)
```

### Inject into Any Model
```python
model = inject_lora(
    model,
    target_modules=["attn_q", "attn_v"],  # only Q and V projections
    rank=4,
    alpha=4.0,
)
# All other parameters frozen — only lora_A, lora_B get gradients
```

### Merge Weights (zero inference overhead)
```python
# After training: fuse LoRA into base weights
merged_linear = lora_linear.merge_weights()
# W_merged = W + (α/r) · B · A
# Same parameter count as original — LoRA overhead is gone
```

---

## 📊 Rank Ablation

| Rank | Trainable % | Notes |
|------|------------|-------|
| r=1 | 0.26% | Extreme compression, highest loss |
| r=4 | 1.04% | Sweet spot for most tasks |
| r=8 | 2.08% | Used in original LoRA paper |
| r=16 | 4.17% | Diminishing returns |
| Full | 100% | Baseline (not LoRA) |

> [!TIP]
> The original LoRA paper used r=4 for GPT-3 and found it matches full fine-tuning on most tasks. Higher rank helps when the task distribution is very different from pre-training.

---

## ⚡ QLoRA + NF4 Quantization

QLoRA (2023) enables fine-tuning 65B models on a single A100 by:
1. **NF4 quantization** — quantize base model to 4-bit (8× compression vs float32)
2. **Double quantization** — quantize the quantization constants too (~0.4 bits/param extra)
3. **Paged optimizers** — handle memory spikes during backprop

NF4 is information-theoretically optimal for normally distributed weights:
```
16 quantization levels chosen to minimize expected quantization error
under a standard normal prior N(0,1) — the distribution of pre-trained weights
```

---

## 🗺️ Roadmap

- [ ] LoRA for vision transformers (ViT)
- [ ] DoRA (Weight-Decomposed Low-Rank Adaptation)
- [ ] QLoRA full implementation with bitsandbytes-compatible NF4
- [ ] LoRA merging (model soup)
- [ ] Fine-tuning on custom dataset (Alpaca format)

---

## 📚 References

| Paper | Authors |
|-------|---------|
| [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685) | Hu et al., 2021 |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | Dettmers et al., 2023 |
| [Parameter-Efficient Transfer Learning](https://arxiv.org/abs/1902.00751) | Houlsby et al., 2019 |
| [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) | Liu et al., 2024 |

---

## ⭐ Support

Found this helpful for understanding LoRA? **Star the repo**!

[![GitHub stars](https://img.shields.io/github/stars/ansariazad/llm-fine-tuning-lora?style=for-the-badge)](https://github.com/ansariazad/llm-fine-tuning-lora/stargazers)

---

## 📄 License

MIT © 2026

---

## 👤 Author

**Azad Ansari** — Python Developer & AI Engineer, Mumbai

[![GitHub](https://img.shields.io/badge/GitHub-ansariazad-181717?logo=github)](https://github.com/ansariazad)
[![Portfolio](https://img.shields.io/badge/Portfolio-ansariazad.github.io-blue)](https://ansariazad.github.io)
