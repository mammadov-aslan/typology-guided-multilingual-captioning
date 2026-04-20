"""
models.py — Core model components for typology-guided multilingual captioning.

Architecture overview:
    BLIP-2 (frozen) → ProjectionMLP → [optional: FiLM / TypologyPrompts] → mT5 + LoRA
"""

import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType


# ── Projection Layer ──────────────────────────────────────────────────────────

class ProjectionMLP(nn.Module):
    """
    Maps frozen BLIP-2 Q-Former output to mT5 encoder input space.

    Input:  (B, 32, 768)  — 32 query tokens from Q-Former
    Output: (B, 32, 768)  — projected visual tokens
    Params: ~1.2M
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── FiLM Typological Conditioning (O2) ───────────────────────────────────────

class FiLMGenerator(nn.Module):
    """
    Generates per-dimension FiLM scale (γ) and shift (β) from a URIEL
    typological vector, following Perez et al. (2018).

    FiLM modulation:  F̃ = γ ⊙ F + β
    where F ∈ R^(B×32×768) are projected visual tokens.

    The final layer is zero-initialised so that at epoch 0 the operation
    is an exact identity (γ=1, β=0), matching the O1-lang baseline.

    Input:  (B, 103)  — URIEL typological vector
    Output: γ, β each (B, 768), broadcast across the 32 sequence positions
    Params: ~211K
    """

    def __init__(self, uriel_dim: int = 103, hidden_dim: int = 128, feat_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(uriel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim * 2),   # outputs [γ | β]
        )
        # Identity initialisation: γ=1, β=0 at epoch 0
        nn.init.zeros_(self.net[-1].weight)
        init_bias = torch.zeros(feat_dim * 2)
        init_bias[:feat_dim] = 1.0                 # γ starts at 1
        self.net[-1].bias = nn.Parameter(init_bias)

        self.feat_dim = feat_dim

    def forward(self, uriel: torch.Tensor):
        out = self.net(uriel)                       # (B, feat_dim*2)
        gamma, beta = out[:, :self.feat_dim], out[:, self.feat_dim:]
        return gamma, beta


def apply_film(features: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
               gamma_max: float = 3.0, beta_max: float = 1.55) -> torch.Tensor:
    """
    Apply FiLM modulation with principled clamping to prevent out-of-distribution
    feature scaling.

    Bound derivation: projected feature std ≈ 1.551 on XM3600.
    Clamping γ ∈ [0, 3.0] keeps modulated std within the original 3σ range.

    Args:
        features:  (B, 32, 768) projected visual tokens
        gamma:     (B, 768) scale parameters
        beta:      (B, 768) shift parameters
        gamma_max: upper bound for γ (default: 3.0)
        beta_max:  abs bound for β (default: 1.55 ≈ feature std)
    """
    gamma = gamma.clamp(0.0, gamma_max)
    beta  = beta.clamp(-beta_max, beta_max)
    # Broadcast (B, 768) → (B, 32, 768)
    return gamma.unsqueeze(1) * features + beta.unsqueeze(1)


# ── Typology-Based Prompting (O3) ─────────────────────────────────────────────

class TypologyPromptGenerator(nn.Module):
    """
    Maps a URIEL typological vector to k learnable prompt tokens prepended
    to the encoder input sequence.

    Encoder input with prompts:
        enc = [p_1 ... p_k | e_lang | proj(F)] ∈ R^(B × (k+1+32) × 768)

    Prompt tokens serve as additional key-value memory in the decoder's
    cross-attention, allowing typological conditioning without modifying
    the underlying visual feature distribution (unlike FiLM).

    Input:  (B, 103)       — URIEL typological vector
    Output: (B, k, 768)    — prompt tokens
    Params: ~1.6M (k=8)
    """

    def __init__(self, uriel_dim: int = 103, hidden_dim: int = 256,
                 feat_dim: int = 768, n_prompts: int = 8):
        super().__init__()
        self.n_prompts = n_prompts
        self.net = nn.Sequential(
            nn.Linear(uriel_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_prompts * feat_dim),
        )

    def forward(self, uriel: torch.Tensor) -> torch.Tensor:
        B = uriel.size(0)
        tokens = self.net(uriel)                    # (B, n_prompts * feat_dim)
        return tokens.view(B, self.n_prompts, -1)   # (B, n_prompts, feat_dim)


# ── mT5 with LoRA ─────────────────────────────────────────────────────────────

def load_mt5_with_lora(model_name: str = "google/mt5-base",
                       r: int = 16, lora_alpha: int = 32,
                       lora_dropout: float = 0.1) -> nn.Module:
    """
    Load mT5-base and attach LoRA adapters to query and value projection
    matrices. All other parameters remain frozen.

    Trainable params: ~7M out of 580M total (~1.2%).
    """
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
