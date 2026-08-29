"""Train an interpretable Logistic Regression pipeline.

Pipeline
--------
1. Load synthetic CSV
2. Engineer features
3. Stratified train/test split (80/20)
4. StandardScaler → LogisticRegression
5. Evaluate (ROC-AUC, precision, recall, F1, confusion matrix)
6. Cross-validate
7. Save pipeline with joblib

Model Choice Rationale
----------------------
Logistic Regression was selected because:
* Coefficients are directly interpretable — each feature's contribution
  to the log-odds can be read off and explained to stakeholders.
* It performs well on structured, low-dimensional tabular data.
* It trains in seconds, making iteration fast.
* Deep learning is not justified for 10 engineered features.

SYNTHETIC DATA DISCLAIMER
--------------------------
All metrics reported here describe model behaviour on *synthetic* data.
They do NOT prove real-world predictive performance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    CV_FOLDS,
    MODEL_FEATURES,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    logger,
)
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
)


def _engineer_row(row: pd.Series) -> dict:
    """Engineer features for a single row."""
    raw = row.to_dict()
    er = compute_engagement_rate(raw)
    return {
        "engagement_rate": er,
        "like_rate": compute_like_rate(raw),
        "comment_rate": compute_comment_rate(raw),
        "view_follower_ratio": compute_view_follower_ratio(raw),
        "category_fit": compute_category_fit(raw),
        "gcc_market_fit": compute_gcc_market_fit(raw),
        "audience_quality": compute_audience_quality(raw),
        "engagement_quality": compute_engagement_quality(raw, er),
        "sponsorship_penalty": compute_sponsorship_penalty(raw),
        "content_consistency": raw["content_consistency_score"],
    }


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer model features for every row in the dataframe."""
    records = [_engineer_row(row) for _, row in df.iterrows()]
    return pd.DataFrame(records)


def train(data_path: Path | None = None) -> Pipeline:
    """Train the scoring pipeline and save it.

    Returns the fitted Pipeline.
    """
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "synthetic_candidates.csv"

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset shape: %s  |  Success rate: %.1f%%",
                df.shape, df["success"].mean() * 100)

    # Engineer features
    X = prepare_features(df)[MODEL_FEATURES]
    y = df["success"]

    # Stratified split ensures the proportion of successful/unsuccessful candidates
    # remains the same in both training and testing sets, preventing class imbalance skew.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Pipeline: StandardScaler ensures all features have zero mean and unit variance.
    # This is critical for Logistic Regression, as it prevents features with naturally
    # larger numerical ranges from dominating the optimization objective.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000,
            solver="lbfgs",
        )),
    ])

    pipeline.fit(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION (on synthetic test set)")
    print("=" * 60)
    print(f"\nROC-AUC : {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    print(f"Cross-Validation ROC-AUC ({CV_FOLDS}-fold): "
          f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature coefficients
    coefs = pipeline.named_steps["classifier"].coef_[0]
    print("\nFeature Coefficients:")
    for name, coef in sorted(zip(MODEL_FEATURES, coefs), key=lambda x: -abs(x[1])):
        print(f"  {name:30s} {coef:+.4f}")

    print("\n⚠  DISCLAIMER: These metrics describe prototype behaviour on")
    print("   synthetic data only. They do NOT prove real-world performance.")
    print("=" * 60)

    # ── Save ───────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Pipeline saved → %s", MODEL_PATH)

    return pipeline


if __name__ == "__main__":
    train()
