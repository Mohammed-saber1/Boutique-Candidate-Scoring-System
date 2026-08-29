"""Tests for the recommendation engine.

Verifies threshold logic and action messages.
"""

import pytest

from app.schemas.candidate import Recommendation
from app.scoring.recommender import recommend


class TestRecommend:
    def test_onboard_at_threshold(self):
        rec, action = recommend(70)
        assert rec == Recommendation.APPROVE
        assert "Prioritize" in action

    def test_onboard_high(self):
        rec, _ = recommend(95)
        assert rec == Recommendation.APPROVE

    def test_hold_at_lower_threshold(self):
        rec, action = recommend(50)
        assert rec == Recommendation.REVIEW
        assert "deeper audience analytics" in action

    def test_hold_mid(self):
        rec, _ = recommend(65)
        assert rec == Recommendation.REVIEW

    def test_hold_just_below_onboard(self):
        rec, _ = recommend(69)
        assert rec == Recommendation.REVIEW

    def test_pass_below_threshold(self):
        rec, action = recommend(49)
        assert rec == Recommendation.DECLINE
        assert "Do not prioritize" in action

    def test_pass_zero(self):
        rec, _ = recommend(0)
        assert rec == Recommendation.DECLINE

    def test_pass_low(self):
        rec, _ = recommend(25)
        assert rec == Recommendation.DECLINE

    def test_perfect_score(self):
        rec, _ = recommend(100)
        assert rec == Recommendation.APPROVE

    @pytest.mark.parametrize("score", range(0, 101, 5))
    def test_always_returns_valid_recommendation(self, score):
        rec, action = recommend(score)
        assert rec in Recommendation
        assert isinstance(action, str)
        assert len(action) > 10  # non-trivial message
