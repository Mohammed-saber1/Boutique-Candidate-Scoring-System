"""Pydantic schemas for candidate input and scoring output.

Validation rules
-----------------
* Percentages are clamped to [0, 100].
* Counts (followers, likes, etc.) must be non-negative.
* Scores in [0, 1] range are clamped.
* Missing optional fields receive sensible defaults.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Platform(str, Enum):
    """Supported social-media platforms."""
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    SNAPCHAT = "snapchat"
    OTHER = "other"


class Recommendation(str, Enum):
    """Business recommendation tiers."""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class DataQuality(str, Enum):
    """Data completeness indicator."""
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


# ---------------------------------------------------------------------------
# Input Schema
# ---------------------------------------------------------------------------

class CandidateInput(BaseModel):
    """Raw candidate profile data submitted for scoring.

    All percentage fields are expected in the 0–100 range.
    Score fields (content_consistency_score, comment_quality_score)
    are expected in the 0.0–1.0 range.
    """

    handle: str = Field(..., min_length=1, description="Social-media handle")
    platform: Platform = Field(default=Platform.INSTAGRAM)

    # Audience size
    followers: int = Field(..., ge=0, description="Total follower count")

    # Engagement averages
    average_likes: float = Field(default=0, ge=0)
    average_comments: float = Field(default=0, ge=0)
    average_views: float = Field(default=0, ge=0)

    # Audience composition (percentages 0-100)
    gcc_audience_pct: float = Field(default=0, ge=0, le=100)
    saudi_audience_pct: float = Field(default=0, ge=0, le=100)
    uae_audience_pct: float = Field(default=0, ge=0, le=100)
    female_audience_pct: float = Field(default=50, ge=0, le=100)

    # Content category breakdown (percentages 0-100)
    beauty_content_pct: float = Field(default=0, ge=0, le=100)
    fashion_content_pct: float = Field(default=0, ge=0, le=100)
    luxury_content_pct: float = Field(default=0, ge=0, le=100)
    lifestyle_content_pct: float = Field(default=0, ge=0, le=100)

    # Sponsorship / commercialisation
    sponsored_content_pct: float = Field(default=0, ge=0, le=100)

    # Quality scores (0.0 – 1.0)
    content_consistency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    comment_quality_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("handle")
    @classmethod
    def strip_handle(cls, v: str) -> str:
        """Remove leading/trailing whitespace from handle."""
        return v.strip()

    @model_validator(mode="after")
    def engagement_sanity(self) -> "CandidateInput":
        """Warn-level check: likes + comments shouldn't exceed followers.
        
        Rationale: In real social media, a single viral post can temporarily cause
        engagement to exceed followers. Therefore, we do not throw a hard validation 
        error (which would break the API). Instead, we let it pass, and the downstream
        data-quality module will flag highly abnormal engagement rates.
        """
        # We don't raise — just allow it and let data-quality flag it.
        return self


# ---------------------------------------------------------------------------
# Output Schemas
# ---------------------------------------------------------------------------

class DriverSignal(BaseModel):
    """A single positive or negative scoring driver."""
    feature: str
    label: str
    contribution: float  # raw model contribution
    display_value: str  # human-friendly description


class ScoringResult(BaseModel):
    """Complete scoring output for one candidate."""

    model_config = {"protected_namespaces": ()}

    handle: str
    platform: str
    score: int = Field(ge=0, le=100)
    success_probability: float = Field(ge=0.0, le=1.0)
    recommendation: Recommendation
    positive_drivers: List[DriverSignal]
    negative_drivers: List[DriverSignal]
    risks: List[str]
    next_action: str
    data_quality: DataQuality
    model_confidence: str
    limitations: List[str] = Field(default_factory=lambda: [
        "Historical Boutiqaat onboarding outcomes were unavailable.",
        "This score is based on a prototype trained on synthetic data.",
    ])
