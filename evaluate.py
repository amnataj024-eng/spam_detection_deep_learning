"""Evaluate the trained model on the held-out test set.

Usage:
    python evaluate.py
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)

from src import config as C
from src.predictor import SpamPredictor
from src.visualization import plot_confusion_matrix, plot_roc


def main() -> None:
    if not C.SPLIT_PATH.exists():
        raise SystemExit("No test split found. Run `python train.py` first.")
    test_df = pd.read_csv(C.SPLIT_PATH)
    predictor = SpamPredictor()

    probs = predictor.predict_proba(test_df["text"].tolist())
    y_true = test_df["label"].values
    y_pred = (probs >= C.THRESHOLD).astype(int)

    metrics = {
        "test_samples": int(len(test_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred)),
        "recall_spam": float(recall_score(y_true, y_pred)),
        "f1_spam": float(f1_score(y_true, y_pred)),
    }
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, C.FIGURES_DIR / "confusion_matrix.png")
    metrics["roc_auc"] = plot_roc(y_true, probs, C.FIGURES_DIR / "roc_curve.png")

    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"], digits=4))
    for k, v in metrics.items():
        print(f"{k:>15}: {v:.4f}" if isinstance(v, float) else f"{k:>15}: {v}")

    # Misclassified examples (useful for the error analysis in the report)
    wrong = test_df.assign(spam_probability=probs, predicted=y_pred)
    wrong = wrong[wrong["label"] != wrong["predicted"]]
    wrong.to_csv(C.MODELS_DIR / "misclassified.csv", index=False)
    metrics["misclassified"] = int(len(wrong))

    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {C.METRICS_PATH} and plots to {C.FIGURES_DIR}")


if __name__ == "__main__":
    main()
