"""
dataset.py — Dataset classes and data utilities for multilingual captioning.

Covers:
  - Precomputed Q-Former feature loading
  - Multilingual caption datasets (7 training languages)
  - Balanced WeightedRandomSampler to prevent language dominance
  - XM3600 evaluation dataset loading
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from typing import Optional


# ── Language Configuration ────────────────────────────────────────────────────

TRAINING_LANGUAGES = ["en", "de", "uk", "ar", "tr", "bn", "vi"]

LANG_PREFIXES = {
    "en": "English:",
    "de": "Deutsch:",
    "uk": "Українська:",
    "ar": "العربية:",
    "tr": "Türkçe:",
    "bn": "বাংলা:",
    "vi": "Tiếng Việt:",
}

LANG_TO_ID = {lang: i for i, lang in enumerate(TRAINING_LANGUAGES)}


# ── Precomputed Feature Dataset ───────────────────────────────────────────────

class MultilingualCaptionDataset(Dataset):
    """
    Dataset that loads precomputed BLIP-2 Q-Former features and pairs them
    with tokenised captions for a single language.

    Args:
        features:      (N_images, 32, 768) tensor of precomputed features
        captions:      list of caption strings
        img_indices:   list mapping caption index → image feature index
        lang:          ISO language code (e.g. "de")
        tokenizer:     mT5 tokenizer
        max_length:    maximum token length for captions (default: 64)
    """

    def __init__(self, features: torch.Tensor, captions: list, img_indices: list,
                 lang: str, tokenizer, max_length: int = 64):
        self.features    = features
        self.captions    = captions
        self.img_indices = img_indices
        self.lang        = lang
        self.lang_id     = LANG_TO_ID[lang]
        self.tokenizer   = tokenizer
        self.max_length  = max_length

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        feature = self.features[self.img_indices[idx]]  # (32, 768)
        labels  = self.tokenizer(
            self.captions[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        return feature, labels, torch.tensor(self.lang_id)


def build_multilingual_loaders(feature_files: dict, tokenizer,
                                batch_size: int = 16, val_frac: float = 0.1,
                                seed: int = 42, num_workers: int = 2):
    """
    Build balanced train/val DataLoaders across all 7 training languages.

    Without balancing the sampler, the model converges to generating
    Vietnamese (the largest dataset, 61K samples) regardless of input language.
    WeightedRandomSampler assigns each sample weight inversely proportional
    to its language size, yielding ~14,500 samples per language per epoch.

    Args:
        feature_files: dict mapping lang → path to .pt feature file
                       Each .pt file should contain:
                         {"features": Tensor, "captions": list, "caption_to_img_idx": list}
        tokenizer:     mT5 tokenizer
        batch_size:    training batch size
        val_frac:      fraction of data held out for validation
        seed:          random seed for reproducibility

    Returns:
        train_loader, val_loader
    """
    all_datasets = []

    for lang in TRAINING_LANGUAGES:
        if lang not in feature_files:
            continue
        saved = torch.load(feature_files[lang], weights_only=False)
        ds = MultilingualCaptionDataset(
            features    = saved["features"],
            captions    = saved["captions"],
            img_indices = saved["caption_to_img_idx"],
            lang        = lang,
            tokenizer   = tokenizer,
        )
        all_datasets.append(ds)

    from torch.utils.data import ConcatDataset, random_split
    full_dataset = ConcatDataset(all_datasets)

    n_val   = int(len(full_dataset) * val_frac)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    # Build per-sample weights for balanced sampling
    lang_sizes  = [len(ds) for ds in all_datasets]
    sample_weights = []
    for ds in all_datasets:
        weight = 1.0 / len(ds)
        sample_weights.extend([weight] * len(ds))

    # Only weight training samples
    train_weights = [sample_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(
        weights     = train_weights,
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# ── XM3600 Evaluation Dataset ─────────────────────────────────────────────────

class XM3600Dataset(Dataset):
    """
    Wraps precomputed BLIP-2 features and XM3600 reference captions for
    evaluation on a single language.

    Args:
        features:      (3600, 32, 768) tensor of precomputed features
        image_ids:     list of 3600 image IDs (aligned with features)
        refs:          dict mapping image_id → list of reference captions
        lang:          ISO language code
        lang_prefix:   native-script prefix string (e.g. "Deutsch:")
        tokenizer:     mT5 tokenizer (for forced_bos_token_id lookup)
    """

    def __init__(self, features: torch.Tensor, image_ids: list, refs: dict,
                 lang: str, lang_prefix: str, tokenizer):
        self.features   = features
        self.image_ids  = image_ids
        self.refs       = refs
        self.lang       = lang

        prefix_ids = tokenizer(
            lang_prefix, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        self.forced_bos = prefix_ids[0].item()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        iid     = self.image_ids[idx]
        feature = self.features[idx]
        refs    = self.refs.get(iid, [])
        return feature, iid, refs


def load_xm3600_features(feature_file: str):
    """Load precomputed XM3600 features from disk."""
    saved = torch.load(feature_file, weights_only=False)
    return saved["features"], saved["image_ids"]


def load_xm3600_refs(data_dir: str, lang: str) -> dict:
    """
    Load XM3600 reference captions for a given language from parquet files.

    Returns:
        dict mapping image_id → list of reference caption strings
    """
    from datasets import load_dataset, Image as HFImage

    ds = load_dataset(
        "parquet",
        data_files={lang: os.path.join(data_dir, f"{lang}-*.parquet")},
        split=lang,
    )
    refs = {}
    for sample in ds:
        refs[sample["image_id"]] = sample["captions"]
    return refs
