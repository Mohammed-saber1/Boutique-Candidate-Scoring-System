"""Feature extraction: convert a validated CandidateInput into a flat dict.

This module bridges the Pydantic schema and the feature-engineering layer.
All raw numeric fields are extracted; no derived features are computed here.
"""

from __future__ import annotations

from typing import Dict

from app.schemas.candidate import CandidateInput


def extract_raw_features(candidate: CandidateInput) -> Dict[str, float]:
    """Return a flat dictionary of raw numeric features.

    Parameters
    ----------
    candidate : CandidateInput
        Validated candidate profile.

    Returns
    -------
    dict
        Keys are feature names, values are floats.
    """
    return {
        "followers": float(candidate.followers),
        "average_likes": float(candidate.average_likes),
        "average_comments": float(candidate.average_comments),
        "average_views": float(candidate.average_views),
        "gcc_audience_pct": candidate.gcc_audience_pct,
        "saudi_audience_pct": candidate.saudi_audience_pct,
        "uae_audience_pct": candidate.uae_audience_pct,
        "female_audience_pct": candidate.female_audience_pct,
        "beauty_content_pct": candidate.beauty_content_pct,
        "fashion_content_pct": candidate.fashion_content_pct,
        "luxury_content_pct": candidate.luxury_content_pct,
        "lifestyle_content_pct": candidate.lifestyle_content_pct,
        "sponsored_content_pct": candidate.sponsored_content_pct,
        "content_consistency_score": candidate.content_consistency_score,
        "comment_quality_score": candidate.comment_quality_score,
    }
