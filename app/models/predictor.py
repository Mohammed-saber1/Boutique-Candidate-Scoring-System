"""Model predictor: load the trained pipeline and predict.

This module provides a thin wrapper around the saved sklearn Pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.config import MODEL_FEATURES, MODEL_PATH, logger


_pipeline: Pipeline | None = None


def _load_pipeline(path: Path | None = None) -> Pipeline:
    """Load (and cache) the trained pipeline.
    
    Rationale: We use a global singleton cache (`_pipeline`) so that the web server
    or CLI does not suffer from disk I/O latency (loading the .joblib file) on 
    every single scoring request.
    """
    global _pipeline
    if _pipeline is None:
        p = path or MODEL_PATH
        if not p.exists():
            raise FileNotFoundError(
                f"Model not found at {p}. Run `python -m app.models.train` first."
            )
        _pipeline = joblib.load(p)
        logger.info("Pipeline loaded from %s", p)
    return _pipeline


def predict_probability(features: Dict[str, float]) -> float:
    """Return success probability for a single candidate.

    Parameters
    ----------
    features : dict
        Engineered features (keys must match ``MODEL_FEATURES``).

    Returns
    -------
    float
        Probability in [0, 1].
    """
    pipeline = _load_pipeline()
    
    # Rationale: Scikit-learn Pipelines expect 2D arrays/DataFrames. We explicitly
    # convert the 1D feature dict to a 1-row DataFrame, ensuring column ordering
    # exactly matches what was used during training (MODEL_FEATURES).
    row = pd.DataFrame([{k: features[k] for k in MODEL_FEATURES}])
    
    prob = pipeline.predict_proba(row)[0, 1]
    return float(prob)


def get_pipeline(path: Path | None = None) -> Pipeline:
    """Return the loaded pipeline (for explainability access)."""
    return _load_pipeline(path)


def reset_pipeline() -> None:
    """Clear the cached pipeline (useful for testing)."""
    global _pipeline
    _pipeline = None
