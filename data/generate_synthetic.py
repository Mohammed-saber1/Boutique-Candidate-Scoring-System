"""Generate a synthetic dataset of influencer candidates.

IMPORTANT — SYNTHETIC DATA DISCLAIMER
--------------------------------------
This script creates *synthetic* candidate profiles for development and
demonstration purposes.  The ``success`` labels are generated using a
transparent probabilistic mechanism (documented below) that encodes
reasonable business assumptions about what makes a strong Boutiqaat
Boutique candidate.

The dataset does NOT represent real Boutiqaat onboarding outcomes and
must NOT be used to make claims about real-world predictive performance.

Label-Generation Logic
----------------------
A latent "fit score" is computed as a weighted sum of normalised signals:

    fit = (  0.25 * gcc_market_fit
           + 0.20 * category_fit
           + 0.20 * engagement_quality
           + 0.10 * audience_quality
           + 0.10 * content_consistency_score
           + 0.05 * view_follower_ratio_capped
           - 0.10 * sponsorship_excess )

The fit score is passed through a sigmoid to produce a probability, then
a Bernoulli draw (with added Gaussian noise on the logit) produces the
binary label.  This ensures:
  * The problem is NOT trivially separable.
  * Noise prevents a deterministic threshold from solving it.
  * Stronger GCC fit, category fit, and engagement increase success
    probability — matching business intuition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.features.engineering import (
    compute_category_fit,
    compute_engagement_quality,
    compute_engagement_rate,
    compute_gcc_market_fit,
    compute_audience_quality,
    compute_sponsorship_penalty,
    compute_view_follower_ratio,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(
    n: int = 1000,
    seed: int = 42,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate *n* synthetic candidate profiles.

    Parameters
    ----------
    n : int
        Number of candidates.
    seed : int
        Random seed for reproducibility.
    output_path : Path, optional
        If provided, save to CSV.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)

    # ── Raw signals ────────────────────────────────────────────────────
    followers = rng.lognormal(mean=12.5, sigma=1.2, size=n).astype(int)
    followers = np.clip(followers, 5000, 50_000_000)

    # Engagement scales inversely with follower count (realistic)
    base_er = rng.beta(2, 50, size=n)  # mostly in 0.01–0.08 range
    avg_likes = (followers * base_er * rng.uniform(0.8, 1.2, size=n)).astype(int)
    avg_comments = (avg_likes * rng.uniform(0.01, 0.06, size=n)).astype(int)
    avg_views = (followers * rng.uniform(0.05, 0.40, size=n)).astype(int)

    gcc_audience_pct = np.clip(rng.normal(55, 25, size=n), 0, 100)
    saudi_audience_pct = np.clip(gcc_audience_pct * rng.uniform(0.3, 0.7, size=n), 0, 100)
    uae_audience_pct = np.clip((gcc_audience_pct - saudi_audience_pct) * rng.uniform(0.2, 0.8, size=n), 0, 100)
    female_audience_pct = np.clip(rng.normal(65, 15, size=n), 0, 100)

    beauty_content_pct = np.clip(rng.normal(30, 20, size=n), 0, 100)
    fashion_content_pct = np.clip(rng.normal(25, 15, size=n), 0, 100)
    luxury_content_pct = np.clip(rng.normal(10, 10, size=n), 0, 100)
    lifestyle_content_pct = np.clip(rng.normal(15, 10, size=n), 0, 100)

    sponsored_content_pct = np.clip(rng.exponential(15, size=n), 0, 100)
    content_consistency_score = np.clip(rng.beta(5, 2, size=n), 0, 1)
    comment_quality_score = np.clip(rng.beta(4, 2, size=n), 0, 1)

    # ── Build DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame({
        "handle": [f"@synthetic_creator_{i:04d}" for i in range(n)],
        "platform": rng.choice(["instagram", "tiktok", "youtube", "snapchat"],
                               size=n, p=[0.55, 0.25, 0.12, 0.08]),
        "followers": followers,
        "average_likes": avg_likes,
        "average_comments": avg_comments,
        "average_views": avg_views,
        "gcc_audience_pct": np.round(gcc_audience_pct, 1),
        "saudi_audience_pct": np.round(saudi_audience_pct, 1),
        "uae_audience_pct": np.round(uae_audience_pct, 1),
        "female_audience_pct": np.round(female_audience_pct, 1),
        "beauty_content_pct": np.round(beauty_content_pct, 1),
        "fashion_content_pct": np.round(fashion_content_pct, 1),
        "luxury_content_pct": np.round(luxury_content_pct, 1),
        "lifestyle_content_pct": np.round(lifestyle_content_pct, 1),
        "sponsored_content_pct": np.round(sponsored_content_pct, 1),
        "content_consistency_score": np.round(content_consistency_score, 3),
        "comment_quality_score": np.round(comment_quality_score, 3),
    })

    # ── Engineer features for label generation ─────────────────────────
    # We compute engineered features row-by-row to reuse the same logic
    # that the model uses, avoiding any divergence.
    eng_features = []
    for _, row in df.iterrows():
        raw = row.to_dict()
        er = compute_engagement_rate(raw)
        eng_features.append({
            "gcc_market_fit": compute_gcc_market_fit(raw),
            "category_fit": compute_category_fit(raw),
            "engagement_quality": compute_engagement_quality(raw, er),
            "audience_quality": compute_audience_quality(raw),
            "content_consistency_score": raw["content_consistency_score"],
            "view_follower_ratio": compute_view_follower_ratio(raw),
            "sponsorship_penalty": compute_sponsorship_penalty(raw),
        })

    eng_df = pd.DataFrame(eng_features)

    # ── Latent fit score ───────────────────────────────────────────────
    # Rationale: We linearly combine the scaled features using predefined business 
    # weights to simulate a "perfect" evaluation of the candidate.
    latent = (
        0.25 * eng_df["gcc_market_fit"]
        + 0.20 * eng_df["category_fit"]
        + 0.20 * eng_df["engagement_quality"]
        + 0.10 * eng_df["audience_quality"]
        + 0.10 * eng_df["content_consistency_score"]
        + 0.05 * np.clip(eng_df["view_follower_ratio"], 0, 1)
        - 0.10 * (1.0 - eng_df["sponsorship_penalty"])  # penalty is inverse
    )

    # Scale to logit range and add noise to prevent trivial separation.
    # Rationale: If we just map the latent score to a probability deterministically,
    # the ML model would perfectly memorize the weights. By adding Gaussian noise 
    # to the logit, we simulate the unpredictable, real-world factors (like 
    # negotiation failures or unmeasured off-platform scandals) that affect onboarding.
    logit = (latent - latent.mean()) / (latent.std() + 1e-8) * 1.5
    noise = rng.normal(0, 0.8, size=n)
    prob = _sigmoid(logit + noise)

    df["success"] = rng.binomial(1, prob)

    # Add a synthetic-data flag column
    df["is_synthetic"] = True

    # ── Save ───────────────────────────────────────────────────────────
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Generated {n} synthetic candidates → {output_path}")
        print(f"  Success rate: {df['success'].mean():.1%}")

    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "synthetic_candidates.csv"
    generate_dataset(n=1000, output_path=out)
