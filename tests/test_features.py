"""Tests for feature extraction and engineering.

These tests verify that derived features are calculated correctly
using deterministic inputs with known expected outputs.
"""

import pytest

from app.features.engineering import (
    compute_audience_quality,
    compute_category_fit,
    compute_comment_rate,
    compute_engagement_quality,
    compute_engagement_rate,
    compute_gcc_market_fit,
    compute_like_rate,
    compute_sponsorship_penalty,
    compute_view_follower_ratio,
    engineer_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(overrides: dict | None = None) -> dict:
    """Create a baseline raw-feature dict with sensible defaults."""
    base = {
        "followers": 100_000.0,
        "average_likes": 5_000.0,
        "average_comments": 200.0,
        "average_views": 20_000.0,
        "gcc_audience_pct": 60.0,
        "saudi_audience_pct": 35.0,
        "uae_audience_pct": 15.0,
        "female_audience_pct": 70.0,
        "beauty_content_pct": 40.0,
        "fashion_content_pct": 30.0,
        "luxury_content_pct": 10.0,
        "lifestyle_content_pct": 10.0,
        "sponsored_content_pct": 15.0,
        "content_consistency_score": 0.75,
        "comment_quality_score": 0.80,
    }
    if overrides:
        base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Engagement Rate
# ---------------------------------------------------------------------------

class TestEngagementRate:
    def test_normal(self):
        raw = _make_raw()
        er = compute_engagement_rate(raw)
        expected = (5000 + 200) / 100_000
        assert er == pytest.approx(expected)

    def test_zero_followers(self):
        raw = _make_raw({"followers": 0.0})
        assert compute_engagement_rate(raw) == 0.0

    def test_large_engagement(self):
        raw = _make_raw({"average_likes": 100_000.0, "average_comments": 5_000.0})
        er = compute_engagement_rate(raw)
        assert er == pytest.approx(1.05)


# ---------------------------------------------------------------------------
# Like Rate & Comment Rate
# ---------------------------------------------------------------------------

class TestLikeRate:
    def test_normal(self):
        raw = _make_raw()
        assert compute_like_rate(raw) == pytest.approx(0.05)

    def test_zero_followers(self):
        raw = _make_raw({"followers": 0.0})
        assert compute_like_rate(raw) == 0.0


class TestCommentRate:
    def test_normal(self):
        raw = _make_raw()
        assert compute_comment_rate(raw) == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# View-to-Follower Ratio
# ---------------------------------------------------------------------------

class TestViewFollowerRatio:
    def test_normal(self):
        raw = _make_raw()
        assert compute_view_follower_ratio(raw) == pytest.approx(0.20)

    def test_zero_followers(self):
        raw = _make_raw({"followers": 0.0})
        assert compute_view_follower_ratio(raw) == 0.0


# ---------------------------------------------------------------------------
# Category Fit
# ---------------------------------------------------------------------------

class TestCategoryFit:
    def test_perfect_beauty(self):
        raw = _make_raw({
            "beauty_content_pct": 100.0,
            "fashion_content_pct": 0.0,
            "luxury_content_pct": 0.0,
            "lifestyle_content_pct": 0.0,
        })
        # 100 * 0.35 / 100 = 0.35
        assert compute_category_fit(raw) == pytest.approx(0.35)

    def test_all_categories_equal(self):
        raw = _make_raw({
            "beauty_content_pct": 25.0,
            "fashion_content_pct": 25.0,
            "luxury_content_pct": 25.0,
            "lifestyle_content_pct": 25.0,
        })
        # (25*0.35 + 25*0.30 + 25*0.20 + 25*0.15) / 100 = 25/100 = 0.25
        assert compute_category_fit(raw) == pytest.approx(0.25)

    def test_zero_content(self):
        raw = _make_raw({
            "beauty_content_pct": 0.0,
            "fashion_content_pct": 0.0,
            "luxury_content_pct": 0.0,
            "lifestyle_content_pct": 0.0,
        })
        assert compute_category_fit(raw) == 0.0


# ---------------------------------------------------------------------------
# GCC Market Fit
# ---------------------------------------------------------------------------

class TestGccMarketFit:
    def test_high_gcc(self):
        raw = _make_raw({"gcc_audience_pct": 80.0})
        assert compute_gcc_market_fit(raw) == pytest.approx(0.80)

    def test_zero_gcc(self):
        raw = _make_raw({"gcc_audience_pct": 0.0})
        assert compute_gcc_market_fit(raw) == 0.0


# ---------------------------------------------------------------------------
# Audience Quality
# ---------------------------------------------------------------------------

class TestAudienceQuality:
    def test_high_quality(self):
        raw = _make_raw({
            "gcc_audience_pct": 90.0,
            "female_audience_pct": 80.0,
            "saudi_audience_pct": 55.0,
            "uae_audience_pct": 20.0,
        })
        aq = compute_audience_quality(raw)
        # (90*0.40 + 80*0.25 + 55*0.20 + 20*0.15) / 100
        expected = (36 + 20 + 11 + 3) / 100
        assert aq == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Engagement Quality
# ---------------------------------------------------------------------------

class TestEngagementQuality:
    def test_blended(self):
        raw = _make_raw()
        er = compute_engagement_rate(raw)  # 0.052
        eq = compute_engagement_quality(raw, er)
        # er_capped = min(0.052 / 0.10, 1.0) = 0.52
        # eq = 0.52 * 0.50 + 0.80 * 0.50 = 0.26 + 0.40 = 0.66
        assert eq == pytest.approx(0.66, abs=0.01)


# ---------------------------------------------------------------------------
# Sponsorship Penalty
# ---------------------------------------------------------------------------

class TestSponsorshipPenalty:
    def test_below_threshold(self):
        raw = _make_raw({"sponsored_content_pct": 5.0})
        assert compute_sponsorship_penalty(raw) == 1.0

    def test_at_threshold(self):
        raw = _make_raw({"sponsored_content_pct": 10.0})
        assert compute_sponsorship_penalty(raw) == 1.0

    def test_above_threshold(self):
        raw = _make_raw({"sponsored_content_pct": 55.0})
        penalty = compute_sponsorship_penalty(raw)
        assert 0.7 < penalty < 1.0

    def test_max_sponsored(self):
        raw = _make_raw({"sponsored_content_pct": 100.0})
        penalty = compute_sponsorship_penalty(raw)
        assert penalty == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_returns_all_features(self):
        raw = _make_raw()
        features = engineer_features(raw)
        expected_keys = {
            "engagement_rate", "like_rate", "comment_rate",
            "view_follower_ratio", "category_fit", "gcc_market_fit",
            "audience_quality", "engagement_quality",
            "sponsorship_penalty", "content_consistency",
        }
        assert set(features.keys()) == expected_keys

    def test_content_consistency_passthrough(self):
        raw = _make_raw({"content_consistency_score": 0.92})
        features = engineer_features(raw)
        assert features["content_consistency"] == 0.92

    def test_all_values_finite(self):
        raw = _make_raw()
        features = engineer_features(raw)
        for k, v in features.items():
            assert isinstance(v, float), f"{k} is not float"
            assert v == v, f"{k} is NaN"  # NaN check
