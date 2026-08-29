"""Recommendation engine: map score to business action.

Threshold Rationale
-------------------
Boutiqaat's Celebrity Management has *limited* onboarding capacity,
so the system must be selective.  Thresholds are defined in
``app.config`` and applied here.

| Score Range | Recommendation | Logic |
|-------------|----------------|-------|
| ≥ 70        | **APPROVE**    | Top candidates — high fit signals across GCC audience, category, and engagement. |
| 50 – 69     | **REVIEW**     | Promising but not clear-cut — request additional data or manual review. |
| < 50        | **DECLINE**    | Weak fit — do not allocate limited onboarding effort. |

Each recommendation includes a concrete, actionable next-step message
that Celebrity Management can act on immediately.
"""

from __future__ import annotations

from app.config import HOLD_THRESHOLD, ONBOARD_THRESHOLD
from app.schemas.candidate import Recommendation


# Actionable messages per tier
_ACTIONS = {
    Recommendation.APPROVE: (
        "Prioritize this candidate for Celebrity Management outreach. "
        "Initiate preliminary terms discussion."
    ),
    Recommendation.REVIEW: (
        "Request deeper audience analytics and review recent content "
        "quality before allocating onboarding capacity."
    ),
    Recommendation.DECLINE: (
        "Do not prioritize for onboarding at this stage. "
        "Re-evaluate if the candidate's profile evolves significantly."
    ),
}


def recommend(score: int) -> tuple[Recommendation, str]:
    """Return recommendation tier and actionable next step.

    Parameters
    ----------
    score : int
        Fit score in [0, 100].

    Returns
    -------
    tuple[Recommendation, str]
        (recommendation enum, next-action message)
    """
    if score >= ONBOARD_THRESHOLD:
        rec = Recommendation.APPROVE
    elif score >= HOLD_THRESHOLD:
        rec = Recommendation.REVIEW
    else:
        rec = Recommendation.DECLINE
    return rec, _ACTIONS[rec]
