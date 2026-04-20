"""
evaluate.py — XM3600 evaluation for all model variants.

Computes CIDEr and BLEU-1/4 on XM3600 for seen training languages
and zero-shot transfer languages.

Usage:
    # Evaluate O1-lang baseline
    python evaluate.py --stage multilingual --checkpoint outputs/multilingual_best.pt

    # Evaluate O3 soft-prompt model on all 7 training languages
    python evaluate.py --stage prompt --checkpoint outputs/prompt_best.pt

    # Evaluate on specific languages only
    python evaluate.py --stage multilingual --langs en de ar vi
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from models import ProjectionMLP, FiLMGenerator, TypologyPromptGenerator, apply_film, load_mt5_with_lora
from dataset import TRAINING_LANGUAGES, LANG_PREFIXES, LANG_TO_ID, load_xm3600_features, load_xm3600_refs


XM3600_ALL_LANGS = [
    "en", "de", "fr", "nl", "da", "sv", "no", "it", "pt", "ro", "es",
    "uk", "ru", "pl", "cs", "ar", "he", "tr", "hu", "fi",
    "vi", "id", "th", "mi", "bn", "hi", "te", "zh", "ja", "ko",
    "fa", "sw",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",       choices=["multilingual", "film", "prompt"],
                        default="multilingual")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--feature_file", default="xm3600_features.pt")
    parser.add_argument("--xm3600_dir",  default=os.path.expanduser(
                            "~/.cache/huggingface/datasets/floschne___xm3600/data"))
    parser.add_argument("--uriel_file",  default="uriel_vectors.pt")
    parser.add_argument("--langs",       nargs="+", default=TRAINING_LANGUAGES)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--output_file", default="outputs/xm3600_results.json")
    return parser.parse_args()


@torch.no_grad()
def generate_captions(features: torch.Tensor, models_dict: dict, stage: str,
                      lang: str, forced_bos: int, device: str,
                      batch_size: int = 16) -> list:
    """
    Generate captions for a batch of visual features.

    For seen languages, the language embedding provides steering.
    For zero-shot languages, only forced_bos is used.
    """
    for m in models_dict.values():
        if isinstance(m, nn.Module):
            m.eval()

    preds = []
    lang_id = LANG_TO_ID.get(lang)

    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device)
        B = batch.size(0)

        projected = models_dict["projection"](batch)

        if stage == "film" and lang_id is not None:
            lang_ids = torch.full((B,), lang_id, dtype=torch.long, device=device)
            uriel    = models_dict["uriel_vectors"][lang_ids]
            gamma, beta = models_dict["film_gen"](uriel)
            projected   = apply_film(projected, gamma, beta)
            lang_tok    = models_dict["lang_emb"](lang_ids).unsqueeze(1)
            enc_input   = torch.cat([lang_tok, projected], dim=1)

        elif stage == "prompt" and lang_id is not None:
            lang_ids  = torch.full((B,), lang_id, dtype=torch.long, device=device)
            uriel     = models_dict["uriel_vectors"][lang_ids]
            prompts   = models_dict["prompt_gen"](uriel)
            lang_tok  = models_dict["lang_emb"](lang_ids).unsqueeze(1)
            enc_input = torch.cat([prompts, lang_tok, projected], dim=1)

        elif lang_id is not None:  # multilingual, seen language
            lang_ids  = torch.full((B,), lang_id, dtype=torch.long, device=device)
            lang_tok  = models_dict["lang_emb"](lang_ids).unsqueeze(1)
            enc_input = torch.cat([lang_tok, projected], dim=1)

        else:  # zero-shot: no lang embedding available
            enc_input = projected

        gen_kwargs = dict(inputs_embeds=enc_input, max_new_tokens=50, num_beams=4)
        if forced_bos is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos

        out = models_dict["mt5"].generate(**gen_kwargs)
        preds.extend(out.tolist())

    return preds


def score(preds_dict: dict, refs_dict: dict) -> dict:
    """Compute CIDEr and BLEU-1/2/3/4."""
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu

    cider_score, _ = Cider().compute_score(refs_dict, preds_dict)
    bleu_scores, _ = Bleu(4).compute_score(refs_dict, preds_dict)

    return {
        "cider":  round(cider_score * 100, 2),
        "bleu1":  round(bleu_scores[0] * 100, 2),
        "bleu2":  round(bleu_scores[1] * 100, 2),
        "bleu3":  round(bleu_scores[2] * 100, 2),
        "bleu4":  round(bleu_scores[3] * 100, 2),
    }


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage: {args.stage} | Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    # Build forced_bos token IDs for all eval languages
    lang_bos = {}
    for lang in args.langs:
        prefix = LANG_PREFIXES.get(lang, f"{lang}:")
        ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids[0]
        lang_bos[lang] = ids[0].item()

    # Load models
    projection = ProjectionMLP().to(device)
    mt5        = load_mt5_with_lora().to(device)
    models_dict = {"projection": projection, "mt5": mt5}

    lang_emb = nn.Embedding(len(TRAINING_LANGUAGES), 768).to(device)
    models_dict["lang_emb"] = lang_emb

    if args.stage == "film":
        models_dict["film_gen"]      = FiLMGenerator().to(device)
        models_dict["uriel_vectors"] = torch.load(args.uriel_file, weights_only=True).to(device)

    if args.stage == "prompt":
        models_dict["prompt_gen"]    = TypologyPromptGenerator().to(device)
        models_dict["uriel_vectors"] = torch.load(args.uriel_file, weights_only=True).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=True)
    projection.load_state_dict(ckpt["projection"])
    mt5.load_state_dict(ckpt["lora"])
    lang_emb.load_state_dict(ckpt["lang_emb"])
    if "film_gen"   in ckpt: models_dict["film_gen"].load_state_dict(ckpt["film_gen"])
    if "prompt_gen" in ckpt: models_dict["prompt_gen"].load_state_dict(ckpt["prompt_gen"])
    print(f"Loaded: {args.checkpoint}")

    # Load XM3600 features (shared across languages)
    features, image_ids = load_xm3600_features(args.feature_file)
    print(f"XM3600: {len(image_ids)} images")

    # Evaluate per language
    results = {}

    for lang in args.langs:
        print(f"\n── {lang} {'(seen)' if lang in TRAINING_LANGUAGES else '(zero-shot)'} " + "─"*40)

        refs = load_xm3600_refs(args.xm3600_dir, lang)
        forced_bos = lang_bos.get(lang)

        preds = generate_captions(
            features, models_dict, args.stage, lang, forced_bos,
            device, args.batch_size
        )

        preds_dict = {}
        refs_dict  = {}

        for i, iid in enumerate(image_ids):
            if iid not in refs or not refs[iid]:
                continue
            pred_str = tokenizer.decode(preds[i], skip_special_tokens=True)
            preds_dict[i] = [pred_str]
            refs_dict[i]  = refs[iid]

        metrics = score(preds_dict, refs_dict)
        results[lang] = {**metrics, "n": len(preds_dict),
                         "seen": lang in TRAINING_LANGUAGES}

        print(f"  CIDEr: {metrics['cider']:.2f} | "
              f"BLEU-1: {metrics['bleu1']:.2f} | "
              f"BLEU-4: {metrics['bleu4']:.2f}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Lang':<6} {'CIDEr':>7} {'BLEU-1':>7} {'BLEU-4':>7}  {'Status'}")
    print("-" * 60)
    for lang in args.langs:
        r = results[lang]
        print(f"{lang:<6} {r['cider']:>7.2f} {r['bleu1']:>7.2f} {r['bleu4']:>7.2f}  "
              f"{'seen' if r['seen'] else 'zero-shot'}")

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {args.output_file}")


if __name__ == "__main__":
    main()
