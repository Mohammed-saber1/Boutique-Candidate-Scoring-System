"""Scorer: convert model probability to a human-friendly 0–100 score.

The conversion is deliberately simple and transparent:
    score = round(probability × 100)
    score = clamp(score, 0, 100)

We keep model probability separate from business score so that:
* Probability can be used for calibration analysis
* Score is the user-facing number
"""

from __future__ import annotations


def probability_to_score(probability: float) -> int:
    """Convert a [0, 1] probability to a [0, 100] integer score.

    Parameters
    ----------
    probability : float
        Model-predicted success probability.

    Returns
    -------
    int
        Clamped score in [0, 100].
    """
    score = round(probability * 100)
    return max(0, min(100, score))
