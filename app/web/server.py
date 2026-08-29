"""Flask web server for the Boutique Candidate Scoring GUI.

Provides a lightweight web interface for scoring candidates
without needing the CLI.

Usage:
    python -m app.web.server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import MODEL_PATH, logger
from app.main import score_candidate
from app.schemas.candidate import CandidateInput

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)


@app.route("/")
def index():
    """Serve the main GUI page."""
    return render_template("index.html")


@app.route("/api/score", methods=["POST"])
def api_score():
    """Score a candidate from JSON payload."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Remove internal metadata keys (like `_note` used in sample JSONs) 
        # before passing to Pydantic to prevent validation failures on unknown fields.
        data.pop("_note", None)

        # Validate incoming data using Pydantic. This guarantees that the ML pipeline
        # receives correctly typed, bounded data (e.g. percentages between 0 and 100).
        candidate = CandidateInput(**data)

        # Score
        result = score_candidate(candidate)

        # Build response
        response = {
            "handle": result.handle,
            "platform": result.platform,
            "score": result.score,
            "success_probability": result.success_probability,
            "recommendation": result.recommendation.value,
            "positive_drivers": [
                {
                    "feature": d.feature,
                    "label": d.label,
                    "contribution": d.contribution,
                    "display_value": d.display_value,
                }
                for d in result.positive_drivers
            ],
            "negative_drivers": [
                {
                    "feature": d.feature,
                    "label": d.label,
                    "contribution": d.contribution,
                    "display_value": d.display_value,
                }
                for d in result.negative_drivers
            ],
            "risks": result.risks,
            "next_action": result.next_action,
            "data_quality": result.data_quality.value,
            "model_confidence": result.model_confidence,
            "limitations": result.limitations,
        }
        return jsonify(response)

    except Exception as e:
        # Catch any unexpected errors (or Pydantic validation errors) and return 
        # a 422 Unprocessable Entity status to the frontend.
        logger.exception("Scoring error")
        return jsonify({"error": str(e)}), 422


@app.route("/api/sample/<candidate_type>")
def api_sample(candidate_type: str):
    """Return sample candidate data."""
    if candidate_type == "strong":
        path = PROJECT_ROOT / "data" / "sample_candidate.json"
    elif candidate_type == "weak":
        path = PROJECT_ROOT / "data" / "sample_candidate_weak.json"
    else:
        return jsonify({"error": "Unknown type. Use 'strong' or 'weak'"}), 404

    with open(path) as f:
        data = json.load(f)
    data.pop("_note", None)
    return jsonify(data)


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print("\n✗ Model not found. Run:  python -m app.main --train\n")
        sys.exit(1)

    print("\n🚀  Boutique Scoring GUI → http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
