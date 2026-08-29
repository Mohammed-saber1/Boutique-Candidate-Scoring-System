"""Feature-contribution explanations for Logistic Regression.

How Contributions Are Calculated
---------------------------------
For a Logistic Regression the log-odds of the positive class are:

    logit(p) = β₀ + β₁·x₁ + β₂·x₂ + … + βₙ·xₙ

where βᵢ are learned coefficients and xᵢ are *scaled* feature values
(after StandardScaler).  Each term ``βᵢ·xᵢ`` is the contribution of
feature *i* to the log-odds.

To make contributions comparable and user-friendly we:

1. Compute raw contribution = coefficient × scaled feature value.
2. Normalize: divide each contribution by the sum of all |contributions|
   then multiply by |score − 50|.  The *sign* is preserved from the
   raw log-odds contribution so that positive always means "helped"
   and negative always means "hurt", regardless of overall score.
3. Map technical feature names to business-friendly labels.

This approach is *technically valid* — it decomposes the model's
actual decision — while being accessible to non-technical stakeholders.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from app.config import FEATURE_LABELS, MODEL_FEATURES
from app.models.predictor import get_pipeline
from app.schemas.candidate import DriverSignal


def compute_contributions(
    features: Dict[str, float],
    score: int,
) -> Tuple[List[DriverSignal], List[DriverSignal]]:
    """Compute per-feature contributions for a candidate.

    Parameters
    ----------
    features : dict
        Engineered features (keys match ``MODEL_FEATURES``).
    score : int
        The candidate's fit score (0–100).

    Returns
    -------
    tuple[list[DriverSignal], list[DriverSignal]]
        (positive_drivers, negative_drivers), each sorted by
        descending absolute contribution.
    """
    pipeline = get_pipeline()
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["classifier"]

    # Build a single-row DataFrame in the correct feature order
    row = pd.DataFrame([{k: features[k] for k in MODEL_FEATURES}])

    # Scale the features
    scaled = scaler.transform(row)[0]

    coefficients = clf.coef_[0]

    # Raw contribution per feature (in log-odds space)
    raw_contributions = coefficients * scaled

    # Normalise to a user-friendly scale.
    # We scale magnitudes by the absolute distance from a neutral 50,
    # but preserve the *sign* from the raw log-odds contribution so that
    # positive always means "helped the candidate" and negative means
    # "hurt the candidate", regardless of whether the overall score is
    # above or below 50.
    score_delta = abs(score - 50)
    total_abs = np.sum(np.abs(raw_contributions))

    if total_abs < 1e-8:
        # Edge case: all contributions ~0 → uniform
        normalized = raw_contributions
    else:
        normalized = (raw_contributions / total_abs) * score_delta

    positive: List[DriverSignal] = []
    negative: List[DriverSignal] = []

    for fname, raw_c, norm_c in zip(MODEL_FEATURES, raw_contributions, normalized):
        label = FEATURE_LABELS.get(fname, fname)
        display = _format_display(fname, features[fname], norm_c)
        signal = DriverSignal(
            feature=fname,
            label=label,
            contribution=round(float(norm_c), 1),
            display_value=display,
        )
        if norm_c >= 0:
            positive.append(signal)
        else:
            negative.append(signal)

    # Sort by absolute contribution descending
    positive.sort(key=lambda s: -abs(s.contribution))
    negative.sort(key=lambda s: -abs(s.contribution))

    return positive, negative


def _format_display(feature: str, value: float, contribution: float) -> str:
    """Create a human-friendly description string."""
    label = FEATURE_LABELS.get(feature, feature)
    sign = "+" if contribution >= 0 else ""
    # Format value based on feature type
    if "pct" in feature or feature in ("gcc_market_fit", "audience_quality", "category_fit"):
        val_str = f"{value:.0%}" if value <= 1 else f"{value:.1f}%"
    elif "rate" in feature or "ratio" in feature:
        val_str = f"{value:.2%}"
    else:
        val_str = f"{value:.2f}"
    return f"{label}: {val_str} ({sign}{contribution:.1f})"


def identify_risks(
    features: Dict[str, float],
    candidate_handle: str,
) -> List[str]:
    """Identify business-relevant risk signals.

    These are not model outputs — they are rule-based flags that
    complement the model's statistical assessment.
    """
    risks: List[str] = []

    if features.get("sponsorship_penalty", 1.0) < 0.85:
        risks.append("Sponsored content is relatively high — audience trust may be weaker.")

    if features.get("gcc_market_fit", 0) < 0.30:
        risks.append("GCC audience is below 30% — limited regional reach.")

    if features.get("content_consistency", 0) < 0.50:
        risks.append("Content consistency is low — posting cadence may be unreliable.")

    if features.get("engagement_rate", 0) > 0.15:
        risks.append("Engagement rate is unusually high — consider verifying audience authenticity.")

    if features.get("engagement_quality", 0) < 0.30:
        risks.append("Engagement quality is low — comments may lack substantive interaction.")

    if features.get("category_fit", 0) < 0.20:
        risks.append("Weak beauty/fashion/luxury category alignment for Boutiqaat.")

    # Data-quality caveat (always present for external profiles)
    risks.append("Some audience composition data may be estimated from third-party tools.")

    return risks
