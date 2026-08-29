"""Tests for the scoring layer.

Verifies that probability-to-score conversion is correct and
scores are always clamped to [0, 100].
"""

import pytest

from app.scoring.scorer import probability_to_score


class TestProbabilityToScore:
    def test_zero(self):
        assert probability_to_score(0.0) == 0

    def test_one(self):
        assert probability_to_score(1.0) == 100

    def test_mid(self):
        assert probability_to_score(0.5) == 50

    def test_typical(self):
        assert probability_to_score(0.84) == 84

    def test_rounding_up(self):
        assert probability_to_score(0.755) == 76

    def test_rounding_down(self):
        assert probability_to_score(0.744) == 74

    def test_clamp_above(self):
        assert probability_to_score(1.5) == 100

    def test_clamp_below(self):
        assert probability_to_score(-0.1) == 0

    @pytest.mark.parametrize("prob", [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    def test_always_in_range(self, prob):
        score = probability_to_score(prob)
        assert 0 <= score <= 100
        assert isinstance(score, int)
