"""Boutique Candidate Scoring System — CLI entry point.

Usage
-----
Score a candidate:
    python -m app.main --input data/sample_candidate.json

JSON output:
    python -m app.main --input data/sample_candidate.json --format json

Retrain the model:
    python -m app.main --train

Generate sample report:
    python -m app.main --input data/sample_candidate.json --report reports/sample_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import MODEL_PATH, REPORTS_DIR, logger
from app.explainability.explanations import compute_contributions, identify_risks
from app.features.engineering import engineer_features
from app.features.extraction import extract_raw_features
from app.models.predictor import predict_probability
from app.reporting import generate_report, save_report
from app.schemas.candidate import (
    CandidateInput,
    DataQuality,
    ScoringResult,
)
from app.scoring.recommender import recommend
from app.scoring.scorer import probability_to_score


def _assess_data_quality(candidate: CandidateInput) -> DataQuality:
    """Heuristic data-quality assessment.
    
    Evaluates whether the provided profile data is comprehensive or relying on 
    safe default fallbacks. Missing signals reduce the fidelity of the final score.
    """
    missing_signals = 0
    # Check if absolute engagement metrics are entirely missing
    if candidate.average_views == 0:
        missing_signals += 1
    # Check if demographic data is entirely absent
    if candidate.gcc_audience_pct == 0 and candidate.saudi_audience_pct == 0:
        missing_signals += 1
    # Check if content categorization is missing
    if candidate.beauty_content_pct == 0 and candidate.fashion_content_pct == 0:
        missing_signals += 1
    # Check if qualitative signals are resting precisely on their neutral defaults
    if candidate.content_consistency_score == 0.5 and candidate.comment_quality_score == 0.5:
        missing_signals += 1

    if missing_signals == 0:
        return DataQuality.GOOD
    elif missing_signals <= 2:
        return DataQuality.FAIR
    else:
        return DataQuality.POOR


def _assess_model_confidence(probability: float) -> str:
    """Describe model confidence based on distance from decision boundary."""
    distance = abs(probability - 0.5)
    if distance >= 0.30:
        return "High"
    elif distance >= 0.15:
        return "Medium"
    else:
        return "Low"


def score_candidate(candidate: CandidateInput) -> ScoringResult:
    """Run the full scoring pipeline for one candidate.

    Parameters
    ----------
    candidate : CandidateInput
        Validated candidate profile.

    Returns
    -------
    ScoringResult
        Complete assessment.
    """
    # 1. Extract raw features into a flat dictionary
    raw = extract_raw_features(candidate)

    # 2. Engineer features (calculate ratios, blends, and apply penalties)
    features = engineer_features(raw)

    # 3. Predict the raw success probability (0.0 to 1.0) using the Logistic Regression model
    probability = predict_probability(features)

    # 4. Map the probability to a human-friendly score (0 to 100)
    score = probability_to_score(probability)

    # 5. Explain the score by decomposing the model's coefficients
    positive_drivers, negative_drivers = compute_contributions(features, score)

    # 6. Identify rule-based risks (e.g., highly unusual engagement rates)
    risks = identify_risks(features, candidate.handle)

    # 7. Recommend a business action tier based on predefined thresholds
    recommendation, next_action = recommend(score)

    # 8. Meta assessment of the model and data for transparency
    data_quality = _assess_data_quality(candidate)
    model_confidence = _assess_model_confidence(probability)

    return ScoringResult(
        handle=candidate.handle,
        platform=candidate.platform.value,
        score=score,
        success_probability=round(probability, 4),
        recommendation=recommendation,
        positive_drivers=positive_drivers,
        negative_drivers=negative_drivers,
        risks=risks,
        next_action=next_action,
        data_quality=data_quality,
        model_confidence=model_confidence,
    )


def _print_text(result: ScoringResult) -> None:
    """Pretty-print the assessment to the terminal."""
    print()
    print("=" * 60)
    print("  BOUTIQUE CANDIDATE ASSESSMENT")
    print("=" * 60)
    print(f"  Candidate : {result.handle}")
    print(f"  Platform  : {result.platform.capitalize()}")
    print("-" * 60)
    print(f"  FIT SCORE           : {result.score} / 100")
    print(f"  SUCCESS PROBABILITY : {result.success_probability:.2f}")
    print(f"  RECOMMENDATION      : {result.recommendation.value}")
    print("-" * 60)

    print("\n  KEY POSITIVE SIGNALS")
    for d in result.positive_drivers:
        print(f"    ✓ {d.display_value}")

    print("\n  NEGATIVE SIGNALS")
    for d in result.negative_drivers:
        print(f"    ✗ {d.display_value}")

    print("\n  RISKS")
    for r in result.risks:
        print(f"    ⚠ {r}")

    print(f"\n  RECOMMENDED ACTION")
    print(f"    {result.next_action}")
    print(f"\n  Data Quality     : {result.data_quality.value}")
    print(f"  Model Confidence : {result.model_confidence}")

    print("\n  LIMITATIONS")
    for lim in result.limitations:
        print(f"    • {lim}")
    print("=" * 60)
    print()


def _print_json(result: ScoringResult) -> None:
    """Print machine-readable JSON output."""
    output = {
        "handle": result.handle,
        "platform": result.platform,
        "score": result.score,
        "success_probability": result.success_probability,
        "recommendation": result.recommendation.value,
        "positive_drivers": [
            {"feature": d.feature, "label": d.label, "contribution": d.contribution}
            for d in result.positive_drivers
        ],
        "negative_drivers": [
            {"feature": d.feature, "label": d.label, "contribution": d.contribution}
            for d in result.negative_drivers
        ],
        "risks": result.risks,
        "next_action": result.next_action,
        "data_quality": result.data_quality.value,
        "model_confidence": result.model_confidence,
        "limitations": result.limitations,
    }
    print(json.dumps(output, indent=2))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Boutique Candidate Scoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to candidate JSON file",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--report", "-r",
        type=str,
        help="Path to save Markdown report",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain the model on synthetic data",
    )
    args = parser.parse_args()

    # Handle --train
    if args.train:
        from data.generate_synthetic import generate_dataset
        from app.models.train import train as train_model
        from app.config import DATA_DIR

        csv_path = DATA_DIR / "synthetic_candidates.csv"
        generate_dataset(n=1000, output_path=csv_path)
        train_model(data_path=csv_path)
        print("\n✓ Model trained and saved.")
        if not args.input:
            return

    # Require --input for scoring
    if not args.input:
        parser.error("--input is required (or use --train)")

    # Check model exists
    if not MODEL_PATH.exists():
        logger.error("Model not found at %s", MODEL_PATH)
        print(f"\n✗ Model not found. Run:  python -m app.main --train\n")
        sys.exit(1)

    # Load and validate candidate
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n✗ File not found: {input_path}\n")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    # Remove metadata keys
    data.pop("_note", None)

    try:
        candidate = CandidateInput(**data)
    except Exception as e:
        print(f"\n✗ Validation error: {e}\n")
        sys.exit(1)

    # Score
    result = score_candidate(candidate)

    # Output
    if args.format == "json":
        _print_json(result)
    else:
        _print_text(result)

    # Optional report
    if args.report:
        report_path = Path(args.report)
        save_report(result, report_path)
        print(f"✓ Report saved → {report_path}")


if __name__ == "__main__":
    main()
