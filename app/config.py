"""Central configuration for the Boutique Candidate Scoring System.

All business-critical thresholds, weights, and model parameters are
defined here so they can be audited and adjusted in a single place.
"""

import os
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model path can be overridden via environment variable
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "model.joblib")))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("boutique_scoring")

# ---------------------------------------------------------------------------
# Category Fit Weights
# ---------------------------------------------------------------------------
# Boutiqaat's core categories are beauty and fashion.  Luxury and lifestyle
# are relevant but secondary.  Weights reflect business priority.
CATEGORY_WEIGHTS = {
    "beauty": 0.35,
    "fashion": 0.30,
    "luxury": 0.20,
    "lifestyle": 0.15,
}

# ---------------------------------------------------------------------------
# Audience Quality Weights
# ---------------------------------------------------------------------------
# GCC audience is the primary market.  Female audience matters because
# Boutiqaat's customer base skews heavily female.  Saudi and UAE are the
# two largest GCC markets.
AUDIENCE_QUALITY_WEIGHTS = {
    "gcc_audience_pct": 0.40,
    "female_audience_pct": 0.25,
    "saudi_audience_pct": 0.20,
    "uae_audience_pct": 0.15,
}

# ---------------------------------------------------------------------------
# Engagement Quality Weights
# ---------------------------------------------------------------------------
ENGAGEMENT_QUALITY_WEIGHTS = {
    "engagement_rate": 0.50,
    "comment_quality_score": 0.50,
}

# ---------------------------------------------------------------------------
# Sponsorship Penalty
# ---------------------------------------------------------------------------
# Audiences of over-commercialised creators may have weaker trust.
# We apply a gentle linear penalty that caps at 30% (penalty = 0.30).
SPONSORSHIP_PENALTY_MAX = 0.30
SPONSORSHIP_PENALTY_THRESHOLD = 10.0  # no penalty below 10% sponsored

# ---------------------------------------------------------------------------
# Recommendation Thresholds
# ---------------------------------------------------------------------------
# With limited onboarding capacity, the ONBOARD threshold must be selective.
# 70 targets roughly the top ~30% of viable candidates while not being so
# restrictive that strong candidates are missed.
ONBOARD_THRESHOLD = 70
HOLD_THRESHOLD = 50

# ---------------------------------------------------------------------------
# Model Training Parameters
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Features used by the model (order matters for coefficient alignment)
MODEL_FEATURES = [
    "engagement_rate",
    "like_rate",
    "comment_rate",
    "view_follower_ratio",
    "category_fit",
    "gcc_market_fit",
    "audience_quality",
    "engagement_quality",
    "sponsorship_penalty",
    "content_consistency",
]

# Human-friendly labels for explainability output
FEATURE_LABELS = {
    "engagement_rate": "Engagement Rate",
    "like_rate": "Like Rate",
    "comment_rate": "Comment Rate",
    "view_follower_ratio": "View-to-Follower Ratio",
    "category_fit": "Category Fit (Beauty/Fashion)",
    "gcc_market_fit": "GCC Market Fit",
    "audience_quality": "Audience Quality",
    "engagement_quality": "Engagement Quality",
    "sponsorship_penalty": "Sponsorship Intensity",
    "content_consistency": "Content Consistency",
}
