"""Tests for Pydantic data validation.

Verifies that the CandidateInput schema correctly accepts valid data,
rejects invalid data, and handles missing/edge-case values.
"""

import pytest

from app.schemas.candidate import CandidateInput, Platform


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------

class TestValidInput:
    def test_minimal_valid(self):
        """Only required fields: handle and followers."""
        c = CandidateInput(handle="@test", followers=1000)
        assert c.handle == "@test"
        assert c.followers == 1000
        assert c.platform == Platform.INSTAGRAM  # default

    def test_full_valid(self):
        c = CandidateInput(
            handle="@full_example",
            platform="tiktok",
            followers=500_000,
            average_likes=20_000,
            average_comments=500,
            average_views=100_000,
            gcc_audience_pct=75,
            saudi_audience_pct=45,
            uae_audience_pct=20,
            female_audience_pct=68,
            beauty_content_pct=50,
            fashion_content_pct=25,
            luxury_content_pct=10,
            lifestyle_content_pct=15,
            sponsored_content_pct=12,
            content_consistency_score=0.85,
            comment_quality_score=0.90,
        )
        assert c.platform == Platform.TIKTOK
        assert c.gcc_audience_pct == 75

    def test_handle_stripped(self):
        c = CandidateInput(handle="  @spaces  ", followers=100)
        assert c.handle == "@spaces"

    def test_zero_followers(self):
        c = CandidateInput(handle="@zero", followers=0)
        assert c.followers == 0

    def test_defaults_applied(self):
        c = CandidateInput(handle="@defaults", followers=1000)
        assert c.average_likes == 0
        assert c.average_comments == 0
        assert c.female_audience_pct == 50
        assert c.content_consistency_score == 0.5
        assert c.comment_quality_score == 0.5


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

class TestInvalidInput:
    def test_missing_handle(self):
        with pytest.raises(Exception):
            CandidateInput(followers=1000)

    def test_empty_handle(self):
        with pytest.raises(Exception):
            CandidateInput(handle="", followers=1000)

    def test_negative_followers(self):
        with pytest.raises(Exception):
            CandidateInput(handle="@neg", followers=-100)

    def test_negative_likes(self):
        with pytest.raises(Exception):
            CandidateInput(handle="@neg", followers=1000, average_likes=-1)

    def test_pct_above_100(self):
        with pytest.raises(Exception):
            CandidateInput(handle="@hi", followers=1000, gcc_audience_pct=150)

    def test_pct_below_0(self):
        with pytest.raises(Exception):
            CandidateInput(handle="@lo", followers=1000, gcc_audience_pct=-5)

    def test_score_above_1(self):
        with pytest.raises(Exception):
            CandidateInput(
                handle="@hi", followers=1000, content_consistency_score=1.5,
            )

    def test_score_below_0(self):
        with pytest.raises(Exception):
            CandidateInput(
                handle="@lo", followers=1000, comment_quality_score=-0.1,
            )

    def test_invalid_platform(self):
        with pytest.raises(Exception):
            CandidateInput(handle="@bad", followers=1000, platform="twitter")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_boundary_pct_100(self):
        c = CandidateInput(
            handle="@max", followers=1000, gcc_audience_pct=100,
        )
        assert c.gcc_audience_pct == 100

    def test_boundary_pct_0(self):
        c = CandidateInput(
            handle="@min", followers=1000, gcc_audience_pct=0,
        )
        assert c.gcc_audience_pct == 0

    def test_boundary_score_1(self):
        c = CandidateInput(
            handle="@max", followers=1000, content_consistency_score=1.0,
        )
        assert c.content_consistency_score == 1.0

    def test_boundary_score_0(self):
        c = CandidateInput(
            handle="@min", followers=1000, content_consistency_score=0.0,
        )
        assert c.content_consistency_score == 0.0

    def test_very_large_followers(self):
        c = CandidateInput(handle="@huge", followers=50_000_000)
        assert c.followers == 50_000_000
