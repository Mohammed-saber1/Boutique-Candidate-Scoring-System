"""Human-readable report generation.

Produces a Markdown report summarising a candidate's assessment
including score, recommendation, drivers, risks, and limitations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.schemas.candidate import ScoringResult


def generate_report(result: ScoringResult) -> str:
    """Generate a Markdown report from a ScoringResult.

    Parameters
    ----------
    result : ScoringResult
        Complete scoring output.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# Boutique Candidate Assessment",
        "",
        f"**Candidate:** {result.handle}  ",
        f"**Platform:** {result.platform.capitalize()}",
        "",
        "---",
        "",
        "## FIT SCORE",
        "",
        f"### {result.score} / 100",
        "",
        "## SUCCESS PROBABILITY",
        "",
        f"{result.success_probability:.2f}",
        "",
        "## RECOMMENDATION",
        "",
        f"### {result.recommendation.value}",
        "",
        "---",
        "",
    ]

    # Positive drivers
    lines.append("## KEY POSITIVE SIGNALS")
    lines.append("")
    if result.positive_drivers:
        for d in result.positive_drivers:
            lines.append(f"- {d.display_value}")
    else:
        lines.append("- No strong positive signals identified.")
    lines.append("")

    # Negative drivers
    lines.append("## NEGATIVE SIGNALS")
    lines.append("")
    if result.negative_drivers:
        for d in result.negative_drivers:
            lines.append(f"- {d.display_value}")
    else:
        lines.append("- No notable negative signals.")
    lines.append("")

    # Risks
    lines.append("## RISKS")
    lines.append("")
    for r in result.risks:
        lines.append(f"- {r}")
    lines.append("")

    # Action
    lines.append("## RECOMMENDED ACTION")
    lines.append("")
    lines.append(result.next_action)
    lines.append("")

    # Data quality & confidence
    lines.append("---")
    lines.append("")
    lines.append(f"**Data Quality:** {result.data_quality.value}  ")
    lines.append(f"**Model Confidence:** {result.model_confidence}")
    lines.append("")

    # Limitations
    lines.append("## LIMITATIONS")
    lines.append("")
    for lim in result.limitations:
        lines.append(f"- {lim}")
    lines.append("")

    return "\n".join(lines)


def save_report(result: ScoringResult, path: Path) -> None:
    """Generate and save a report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(result)
    path.write_text(report, encoding="utf-8")
