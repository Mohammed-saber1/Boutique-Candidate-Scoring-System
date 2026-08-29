"""Feature engineering: compute derived features from raw signals.

Every derived feature is documented with its formula and business
rationale.  The output is a dict of the 10 model features defined
in ``app.config.MODEL_FEATURES``.

Feature Catalogue
-----------------
1. **engagement_rate**
   ``(average_likes + average_comments) / followers``
   Overall audience-interaction intensity.

2. **like_rate**
   ``average_likes / followers``
   Passive-engagement signal.

3. **comment_rate**
   ``average_comments / followers``
   Active-engagement signal (comments require more effort than likes).

4. **view_follower_ratio**
   ``average_views / followers``
   Content reach beyond follower base.

5. **category_fit**
   Weighted sum of beauty/fashion/luxury/lifestyle percentages,
   normalised to [0, 1].  Weights are in ``app.config.CATEGORY_WEIGHTS``.

6. **gcc_market_fit**
   Direct mapping of ``gcc_audience_pct`` to [0, 1].
   Boutiqaat operates primarily in GCC markets.

7. **audience_quality**
   Weighted combination of GCC%, female%, Saudi%, UAE% — all normalised.
   See ``app.config.AUDIENCE_QUALITY_WEIGHTS``.

8. **engagement_quality**
   Blends engagement_rate with comment_quality_score so that raw volume
   alone doesn't dominate.

9. **sponsorship_penalty**
   A *negative* signal: higher sponsored-content % → lower value.
   Linear ramp from 1.0 (no penalty) to ``1 - SPONSORSHIP_PENALTY_MAX``
   at 100% sponsored content, with a dead-zone below the threshold.

10. **content_consistency**
    Direct passthrough of ``content_consistency_score``.
"""

from __future__ import annotations

from typing import Dict

from app.config import (
    AUDIENCE_QUALITY_WEIGHTS,
    CATEGORY_WEIGHTS,
    ENGAGEMENT_QUALITY_WEIGHTS,
    SPONSORSHIP_PENALTY_MAX,
    SPONSORSHIP_PENALTY_THRESHOLD,
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide safely, returning 0.0 when denominator is zero."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def compute_engagement_rate(raw: Dict[str, float]) -> float:
    """(average_likes + average_comments) / followers."""
    return _safe_ratio(
        raw["average_likes"] + raw["average_comments"],
        raw["followers"],
    )


def compute_like_rate(raw: Dict[str, float]) -> float:
    """average_likes / followers."""
    return _safe_ratio(raw["average_likes"], raw["followers"])


def compute_comment_rate(raw: Dict[str, float]) -> float:
    """average_comments / followers."""
    return _safe_ratio(raw["average_comments"], raw["followers"])


def compute_view_follower_ratio(raw: Dict[str, float]) -> float:
    """average_views / followers."""
    return _safe_ratio(raw["average_views"], raw["followers"])


def compute_category_fit(raw: Dict[str, float]) -> float:
    """Weighted category fit normalised to [0, 1].

    Formula:
        (beauty * 0.35 + fashion * 0.30 + luxury * 0.20 + lifestyle * 0.15) / 100
    """
    score = (
        raw["beauty_content_pct"] * CATEGORY_WEIGHTS["beauty"]
        + raw["fashion_content_pct"] * CATEGORY_WEIGHTS["fashion"]
        + raw["luxury_content_pct"] * CATEGORY_WEIGHTS["luxury"]
        + raw["lifestyle_content_pct"] * CATEGORY_WEIGHTS["lifestyle"]
    )
    return score / 100.0  # normalise to [0, 1]


def compute_gcc_market_fit(raw: Dict[str, float]) -> float:
    """GCC audience percentage normalised to [0, 1]."""
    return raw["gcc_audience_pct"] / 100.0


def compute_audience_quality(raw: Dict[str, float]) -> float:
    """Weighted blend of audience-composition signals, normalised to [0, 1].

    Uses weights from ``AUDIENCE_QUALITY_WEIGHTS``.
    """
    score = (
        raw["gcc_audience_pct"] * AUDIENCE_QUALITY_WEIGHTS["gcc_audience_pct"]
        + raw["female_audience_pct"] * AUDIENCE_QUALITY_WEIGHTS["female_audience_pct"]
        + raw["saudi_audience_pct"] * AUDIENCE_QUALITY_WEIGHTS["saudi_audience_pct"]
        + raw["uae_audience_pct"] * AUDIENCE_QUALITY_WEIGHTS["uae_audience_pct"]
    )
    return score / 100.0  # normalise to [0, 1]


def compute_engagement_quality(
    raw: Dict[str, float],
    engagement_rate: float,
) -> float:
    """Blend engagement rate with comment quality.

    Both components are in [0, 1] (engagement_rate is typically < 0.15
    for real profiles but can be higher for micro-influencers).  
    
    Rationale: We cap the engagement component at 10% (0.10) to keep the blend 
    balanced. Without this cap, an influencer with a tiny follower count but 
    a 40% engagement rate on a single viral video would artificially dominate 
    the quality score, masking poor comment sentiment.
    """
    er_capped = min(engagement_rate / 0.10, 1.0)  # 10% ER → max
    return (
        er_capped * ENGAGEMENT_QUALITY_WEIGHTS["engagement_rate"]
        + raw["comment_quality_score"] * ENGAGEMENT_QUALITY_WEIGHTS["comment_quality_score"]
    )


def compute_sponsorship_penalty(raw: Dict[str, float]) -> float:
    """Return a value in [1 - MAX_PENALTY, 1.0].

    Below the threshold the creator incurs no penalty (returns 1.0).
    Above it the penalty increases linearly.
    """
    pct = raw["sponsored_content_pct"]
    if pct <= SPONSORSHIP_PENALTY_THRESHOLD:
        return 1.0
    # Linear ramp: at 100% → 1 - MAX_PENALTY
    penalty_fraction = (pct - SPONSORSHIP_PENALTY_THRESHOLD) / (
        100.0 - SPONSORSHIP_PENALTY_THRESHOLD
    )
    return 1.0 - SPONSORSHIP_PENALTY_MAX * penalty_fraction


def engineer_features(raw: Dict[str, float]) -> Dict[str, float]:
    """Compute all 10 model features from raw signals.

    Parameters
    ----------
    raw : dict
        Output of ``extraction.extract_raw_features``.

    Returns
    -------
    dict
        Keys match ``config.MODEL_FEATURES``.
    """
    er = compute_engagement_rate(raw)
    return {
        "engagement_rate": er,
        "like_rate": compute_like_rate(raw),
        "comment_rate": compute_comment_rate(raw),
        "view_follower_ratio": compute_view_follower_ratio(raw),
        "category_fit": compute_category_fit(raw),
        "gcc_market_fit": compute_gcc_market_fit(raw),
        "audience_quality": compute_audience_quality(raw),
        "engagement_quality": compute_engagement_quality(raw, er),
        "sponsorship_penalty": compute_sponsorship_penalty(raw),
        "content_consistency": raw["content_consistency_score"],
    }
