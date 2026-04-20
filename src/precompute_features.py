"""
precompute_features.py — Extract and cache BLIP-2 Q-Former features.

Run this once before training to precompute visual features offline.
Supports both Flickr8k (English pretraining) and XM3600 (evaluation).

Usage:
    # Flickr8k features (for English pretraining)
    python precompute_features.py --dataset flickr8k --output flickr_features.pt

    # XM3600 features (for evaluation — run on a machine with internet first)
    python precompute_features.py --dataset xm3600 --output xm3600_features.pt
"""

import argparse
import gc
import os
import torch
from transformers import Blip2Processor, Blip2Model
from datasets import load_dataset, Image as HFImage
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["flickr8k", "xm3600"], required=True)
    parser.add_argument("--output",     required=True, help="Output .pt file path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--xm3600_dir", default=os.path.expanduser(
                            "~/.cache/huggingface/datasets/floschne___xm3600/data"))
    return parser.parse_args()


def extract_qformer_features(images: list, processor, model, device, batch_size: int):
    """Extract Q-Former features for a list of PIL images."""
    all_features = []

    for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)

        with torch.no_grad():
            vision_out = model.vision_model(
                pixel_values=inputs.pixel_values.half(),
                return_dict=True,
            )
            image_embeds = vision_out.last_hidden_state
            image_attn   = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=device)
            query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            qformer_out  = model.qformer(
                query_embeds          = query_tokens,
                encoder_hidden_states = image_embeds,
                encoder_attention_mask = image_attn,
                return_dict           = True,
            )
            feats = qformer_out.last_hidden_state   # (B, 32, 768)

        all_features.append(feats.cpu().float())

    return torch.cat(all_features, dim=0)


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load BLIP-2
    print("Loading BLIP-2...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model     = Blip2Model.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    if args.dataset == "flickr8k":
        print("Loading Flickr8k...")
        ds       = load_dataset("jxie/flickr8k", split="train")
        images   = [s["image"] for s in ds]
        captions = []
        caption_to_img_idx = []
        for i, s in enumerate(ds):
            for j in range(5):
                captions.append(s[f"caption_{j}"])
                caption_to_img_idx.append(i)
        print(f"  {len(images)} images, {len(captions)} captions")

        features = extract_qformer_features(images, processor, model, device, args.batch_size)
        torch.save({
            "features":           features,
            "captions":           captions,
            "caption_to_img_idx": caption_to_img_idx,
        }, args.output)

    elif args.dataset == "xm3600":
        print("Loading XM3600 (English split for images)...")
        ds_en = load_dataset(
            "parquet",
            data_files={"en": os.path.join(args.xm3600_dir, "en-*.parquet")},
            split="en",
        ).cast_column("image", HFImage())

        images    = [ds_en[i]["image"]    for i in range(len(ds_en))]
        image_ids = [ds_en[i]["image_id"] for i in range(len(ds_en))]
        print(f"  {len(images)} images")

        features = extract_qformer_features(images, processor, model, device, args.batch_size)
        torch.save({
            "features":  features,
            "image_ids": image_ids,
        }, args.output)

    # Free GPU memory
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nSaved {features.shape} features → {args.output}")


if __name__ == "__main__":
    main()
