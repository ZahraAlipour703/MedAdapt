"""Binary classification metrics and patient-level probability aggregation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def aggregate_patient_predictions(
    image_predictions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Average malignant-class probabilities across each patient's images.

    A patient must have one binary ground-truth label. The result contains one
    row per patient, so all patients contribute equally to patient-level metrics
    regardless of their number of images.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_predictions:
        patient = str(row["patient_id"]).strip()
        if not patient:
            raise ValueError("Prediction row contains an empty patient_id.")
        grouped[patient].append(row)

    output = []
    for patient in sorted(grouped):
        rows = grouped[patient]
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(
                f"Patient {patient!r} has conflicting labels in predictions."
            )
        probabilities = [float(row["probability"]) for row in rows]
        if not all(np.isfinite(probabilities)):
            raise ValueError(f"Patient {patient!r} has non-finite probabilities.")
        if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
            raise ValueError(f"Patient {patient!r} has probabilities outside [0, 1].")

        output.append({
            "patient_id": patient,
            "label": labels.pop(),
            "probability": float(np.mean(probabilities)),
            "image_count": len(rows),
        })
    return output


def binary_classification_metrics(
    predictions: Iterable[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Calculate thresholded and ranking metrics for binary labels.

    The probability field is the predicted probability of malignant class 1.
    Average precision is reported as the PR-AUC summary used by scikit-learn.
    ROC-AUC and average precision are None when the evaluated set has one class.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    rows = list(predictions)
    if not rows:
        raise ValueError("Cannot calculate metrics for an empty prediction set.")

    y_true = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    y_prob = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("All labels must be binary (0 or 1).")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")

    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    positive_support = int((y_true == 1).sum())
    negative_support = int((y_true == 0).sum())
    predicted_positive = int((y_pred == 1).sum())
    sensitivity = float(tp / positive_support) if positive_support else None
    specificity = float(tn / negative_support) if negative_support else None
    both_classes = positive_support > 0 and negative_support > 0
    balanced_accuracy = (
        float((sensitivity + specificity) / 2.0)
        if both_classes else None
    )

    return {
        "n_samples": int(len(rows)),
        "n_benign": int((y_true == 0).sum()),
        "n_malignant": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(tp / predicted_positive) if predicted_positive else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(y_true, y_prob)) if both_classes else None
        ),
        "log_loss": float(log_loss(
            y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]
        )),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
