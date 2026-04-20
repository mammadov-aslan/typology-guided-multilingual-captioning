"""
train.py — Training script for all model variants.

Supports three training stages:
  Stage 0:  English pretraining on Flickr8k (ProjectionMLP + LoRA)
  Stage 1:  Multilingual baseline (O1-lang) — adds language embedding
  Stage 2:  Typological conditioning (O2: FiLM, O3: soft prompts)

Usage:
    # Stage 0 — English pretraining
    python train.py --stage english --feature_file flickr_features.pt

    # Stage 1 — Multilingual baseline
    python train.py --stage multilingual --checkpoint outputs/english_best.pt

    # Stage 2a — FiLM conditioning
    python train.py --stage film --checkpoint outputs/multilingual_best.pt

    # Stage 2b — Typology prompting
    python train.py --stage prompt --checkpoint outputs/multilingual_best.pt
"""

import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from models import ProjectionMLP, FiLMGenerator, TypologyPromptGenerator, apply_film, load_mt5_with_lora
from dataset import TRAINING_LANGUAGES, LANG_TO_ID, build_multilingual_loaders


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",       choices=["english", "multilingual", "film", "prompt"],
                        default="english")
    parser.add_argument("--checkpoint",  default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--feature_file", default="flickr_features.pt")
    parser.add_argument("--uriel_file",  default="uriel_vectors.pt",
                        help="(B, 103) URIEL vectors keyed by language code")
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


# ── Training utilities ────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_padding(labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    masked = labels.clone()
    masked[masked == pad_id] = -100
    return masked


def train_one_epoch(models_dict: dict, loader, optimizer, tokenizer, device, stage: str):
    for m in models_dict.values():
        m.train()

    total_loss, n_batches = 0.0, 0

    for features, labels, lang_ids in tqdm(loader, desc="Train", leave=False):
        features = features.to(device)
        labels   = labels.to(device)
        lang_ids = lang_ids.to(device)

        projected = models_dict["projection"](features)           # (B, 32, 768)

        if stage == "film":
            uriel  = models_dict["uriel_vectors"][lang_ids]       # (B, 103)
            gamma, beta = models_dict["film_gen"](uriel)
            projected   = apply_film(projected, gamma, beta)
            enc_input   = torch.cat([
                models_dict["lang_emb"](lang_ids).unsqueeze(1),
                projected
            ], dim=1)                                              # (B, 33, 768)

        elif stage == "prompt":
            uriel    = models_dict["uriel_vectors"][lang_ids]
            prompts  = models_dict["prompt_gen"](uriel)           # (B, 8, 768)
            enc_input = torch.cat([
                prompts,
                models_dict["lang_emb"](lang_ids).unsqueeze(1),
                projected,
            ], dim=1)                                              # (B, 41, 768)

        elif stage == "multilingual":
            enc_input = torch.cat([
                models_dict["lang_emb"](lang_ids).unsqueeze(1),
                projected,
            ], dim=1)                                              # (B, 33, 768)

        else:  # english
            enc_input = projected                                  # (B, 32, 768)

        loss = models_dict["mt5"](
            inputs_embeds = enc_input,
            labels        = mask_padding(labels, tokenizer.pad_token_id),
        ).loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for m in models_dict.values() if hasattr(m, "parameters")
             for p in (m.parameters() if isinstance(m, nn.Module) else [])],
            max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(models_dict: dict, loader, tokenizer, device, stage: str):
    for m in models_dict.values():
        if isinstance(m, nn.Module):
            m.eval()

    total_loss, n_batches = 0.0, 0

    for features, labels, lang_ids in tqdm(loader, desc="Val", leave=False):
        features = features.to(device)
        labels   = labels.to(device)
        lang_ids = lang_ids.to(device)

        projected = models_dict["projection"](features)

        if stage == "film":
            uriel  = models_dict["uriel_vectors"][lang_ids]
            gamma, beta = models_dict["film_gen"](uriel)
            projected   = apply_film(projected, gamma, beta)
            enc_input   = torch.cat([models_dict["lang_emb"](lang_ids).unsqueeze(1), projected], dim=1)

        elif stage == "prompt":
            uriel    = models_dict["uriel_vectors"][lang_ids]
            prompts  = models_dict["prompt_gen"](uriel)
            enc_input = torch.cat([prompts, models_dict["lang_emb"](lang_ids).unsqueeze(1), projected], dim=1)

        elif stage == "multilingual":
            enc_input = torch.cat([models_dict["lang_emb"](lang_ids).unsqueeze(1), projected], dim=1)

        else:
            enc_input = projected

        loss = models_dict["mt5"](
            inputs_embeds = enc_input,
            labels        = mask_padding(labels, tokenizer.pad_token_id),
        ).loss

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Stage: {args.stage} | Device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    # Core models
    projection = ProjectionMLP().to(device)
    mt5        = load_mt5_with_lora().to(device)

    # Stage-specific components
    models_dict = {"projection": projection, "mt5": mt5}

    if args.stage in ("multilingual", "film", "prompt"):
        lang_emb = nn.Embedding(len(TRAINING_LANGUAGES), 768).to(device)
        models_dict["lang_emb"] = lang_emb

    if args.stage == "film":
        film_gen = FiLMGenerator().to(device)
        uriel_vecs = torch.load(args.uriel_file, weights_only=True)  # (n_langs, 103)
        models_dict["film_gen"]      = film_gen
        models_dict["uriel_vectors"] = uriel_vecs.to(device)

    if args.stage == "prompt":
        prompt_gen = TypologyPromptGenerator().to(device)
        uriel_vecs = torch.load(args.uriel_file, weights_only=True)
        models_dict["prompt_gen"]    = prompt_gen
        models_dict["uriel_vectors"] = uriel_vecs.to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=True)
        projection.load_state_dict(ckpt["projection"])
        mt5.load_state_dict(ckpt["lora"])
        if "lang_emb" in ckpt and "lang_emb" in models_dict:
            models_dict["lang_emb"].load_state_dict(ckpt["lang_emb"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Dataloader
    feature_files = {lang: f"{lang}_features.pt" for lang in TRAINING_LANGUAGES}
    if args.stage == "english":
        feature_files = {"en": args.feature_file}

    train_loader, val_loader = build_multilingual_loaders(
        feature_files = feature_files,
        tokenizer     = tokenizer,
        batch_size    = args.batch_size,
    )

    # Optimizer
    trainable_params = [
        p for m in models_dict.values()
        if isinstance(m, nn.Module)
        for p in m.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(models_dict, train_loader, optimizer, tokenizer, device, args.stage)
        val_loss   = validate(models_dict, val_loader, tokenizer, device, args.stage)
        scheduler.step()

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "projection": projection.state_dict(),
                "lora":       mt5.state_dict(),
            }
            if "lang_emb" in models_dict:
                ckpt["lang_emb"] = models_dict["lang_emb"].state_dict()
            if "film_gen" in models_dict:
                ckpt["film_gen"] = models_dict["film_gen"].state_dict()
            if "prompt_gen" in models_dict:
                ckpt["prompt_gen"] = models_dict["prompt_gen"].state_dict()

            save_path = os.path.join(args.output_dir, f"{args.stage}_best.pt")
            torch.save(ckpt, save_path)
            print(f"  ✓ Saved best checkpoint → {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
