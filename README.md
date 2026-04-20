# Typology-Guided Multilingual Image Captioning for Low Resource Languages

Multilingual image captioning system that conditions generation on **linguistic typology** (URIEL vectors) to improve caption quality for low-resource languages.

---

## Overview

Standard multilingual captioning models degrade significantly on low-resource languages due to data scarcity. This project investigates whether **typological information** (structural language properties such as word order and morphology) can bridge this gap by guiding a vision-language model toward language-appropriate generation.

We build on top of a frozen **BLIP-2** visual encoder and **mT5-base** decoder with LoRA adapters, and introduce typological conditioning through two mechanisms:

| Model | Description |
|-------|-------------|
| **O1-bos** | Decoder-level language steering via forced BOS token |
| **O1-lang** | Encoder-level language embedding (strong baseline) |
| **O2-FiLM** | FiLM modulation of visual features using URIEL vectors |
| **O3-Prompt** | Soft prompt tokens generated from URIEL vectors |

---

## Results (XM3600 Benchmark — CIDEr)

| Model | EN | DE | AR | VI |
|-------|----|----|----|----|
| O1-bos | 9.71 | 3.01 | 2.13 | 4.37 |
| O1-lang | 15.49 | 5.76 | 5.40 | 10.46 |
| O2-FiLM | 14.89 | 5.02 | 4.97 | 10.11 |
| **O3-Prompt** | **16.38** | **6.10** | **5.96** | **10.79** |

Key findings:
- Encoder-level language embedding (O1-lang) is a strong and stable baseline
- Prompt-based typological conditioning yields consistent gains on seen languages (+0.89 CIDEr on EN)
- FiLM conditioning degrades performance due to unconstrained feature scaling (γ grows to 4–5× by epoch 10)
- Zero-shot transfer to unseen languages remains near zero across all models

---

## Architecture

```
Image → BLIP-2 (frozen) → Q-Former → ProjectionMLP
                                           ↓
URIEL vector → FiLMGenerator/PromptGenerator
                                           ↓
                            [Prompts | Lang Embedding | Visual Tokens]
                                           ↓
                                     mT5-base + LoRA → Caption
```

---

## Training Data

| Language | Dataset | Size | Type |
|----------|---------|------|------|
| English | Multi30K | 31K | Crowdsourced |
| German | Multi30K | 31K | Professional MT |
| Ukrainian | Multi30K-uk | 29K | MT + human |
| Arabic | Flickr8k-ar | 24K | MT + human |
| Turkish | TasvirEt | 16K | Human |
| Bengali | BAN-Cap | 40K | Human |
| Vietnamese | UIT-OpenViIC | 61K | Human |

Evaluation: **XM3600** (3,600 images × 36 languages, human-written captions)

---

## Installation

```bash
git clone https://github.com/mammadov-aslan/typology-guided-multilingual-captioning
cd typology-guided-multilingual-captioning
pip install -r requirements.txt
```

---

## Usage

### 1. Precompute visual features (run once)
```bash
python src/precompute_features.py --dataset flickr8k --output flickr_features.pt
python src/precompute_features.py --dataset xm3600   --output xm3600_features.pt
```

### 2. Train

```bash
# Stage 0: English pretraining
python src/train.py --stage english --feature_file flickr_features.pt

# Stage 1: Multilingual baseline (O1-lang)
python src/train.py --stage multilingual --checkpoint outputs/english_best.pt

# Stage 2a: FiLM conditioning (O2)
python src/train.py --stage film --checkpoint outputs/multilingual_best.pt

# Stage 2b: Soft prompt conditioning (O3)
python src/train.py --stage prompt --checkpoint outputs/multilingual_best.pt
```

### 3. Evaluate on XM3600

```bash
python src/evaluate.py --stage prompt \
    --checkpoint outputs/prompt_best.pt \
    --langs en de ar tr bn vi
```

---

## Repository Structure

```
├── src/
│   ├── models.py               # ProjectionMLP, FiLMGenerator, TypologyPromptGenerator
│   ├── dataset.py              # Multilingual datasets, balanced sampling, XM3600 loader
│   ├── train.py                # Training loop (all stages)
│   ├── evaluate.py             # XM3600 evaluation with CIDEr + BLEU
│   └── precompute_features.py  # Offline BLIP-2 feature extraction
│
├── notebooks/
│   ├── 01_english_pretraining.ipynb
│   ├── 02_multilingual_baseline.ipynb
│   └── 03_film_prompt_conditioning.ipynb
│
├── requirements.txt
└── .gitignore
```

---

## Tech Stack

Python · PyTorch · HuggingFace Transformers · PEFT (LoRA) · BLIP-2 · mT5 · URIEL

---

## References

- Li et al. (2023) — [BLIP-2](https://arxiv.org/abs/2301.12597)
- Xue et al. (2021) — [mT5](https://arxiv.org/abs/2010.11934)
- Hu et al. (2022) — [LoRA](https://arxiv.org/abs/2106.09685)
- Perez et al. (2018) — [FiLM](https://arxiv.org/abs/1709.07871)
- Littell et al. (2017) — [URIEL](https://aclanthology.org/E17-2002/)
- Thapliyal et al. (2022) — [XM3600](https://aclanthology.org/2022.emnlp-main.45/)
