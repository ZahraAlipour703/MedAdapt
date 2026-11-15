"""Patient-grouped cross-validation split generation for BreakHis."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from sklearn.model_selection import StratifiedGroupKFold


def generate_grouped_folds(
    rows: Iterable[dict[str, Any]],
    n_splits: int = 5,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Assign every image to train/test sets in patient-disjoint folds.

    Required columns: image_path, label, patient_id and parse_status.
    Each patient must have exactly one binary label. Every patient appears
    in the test set of exactly one fold.
    """
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("The manifest contains no image records.")

    required = {"image_path", "label", "patient_id", "parse_status"}
    missing_columns = required.difference(records[0])
    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    invalid = [
        row for row in records
        if row.get("parse_status") != "ok"
        or not str(row.get("patient_id", "")).strip()
        or str(row.get("label", "")).strip() not in {"0", "1"}
    ]
    if invalid:
        raise ValueError(
            f"{len(invalid)} manifest rows have missing or invalid patient "
            "metadata. Resolve the audit errors before generating folds."
        )

    records.sort(
        key=lambda row: (
            str(row["patient_id"]),
            str(row["image_path"]),
        )
    )

    patient_labels: dict[str, int] = {}
    for row in records:
        patient_id = str(row["patient_id"])
        label = int(row["label"])
        previous = patient_labels.setdefault(patient_id, label)
        if previous != label:
            raise ValueError(
                f"Patient {patient_id!r} has conflicting binary labels "
                f"({previous} and {label}). Check filename parsing and IDs."
            )

    unique_patients = sorted(patient_labels)
    class_patient_counts = {
        label: sum(value == label for value in patient_labels.values())
        for label in (0, 1)
    }
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if min(class_patient_counts.values()) < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} stratified folds: patient counts by "
            f"class are {class_patient_counts}. Each class needs at least "
            f"{n_splits} patients."
        )

    labels = [int(row["label"]) for row in records]
    groups = [str(row["patient_id"]) for row in records]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    output: list[dict[str, Any]] = []
    test_patients_seen: list[set[str]] = []

    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(X=[0] * len(records), y=labels, groups=groups),
        start=1,
    ):
        train_index_set = set(int(index) for index in train_indices)
        test_index_set = set(int(index) for index in test_indices)
        train_patients = {groups[index] for index in train_index_set}
        test_patients = {groups[index] for index in test_index_set}

        if train_patients.intersection(test_patients):
            raise RuntimeError(f"Patient leakage detected in fold {fold_number}.")

        test_patients_seen.append(test_patients)
        for index, row in enumerate(records):
            split_name = "train" if index in train_index_set else "test"
            output.append({**row, "fold": fold_number, "split": split_name})

    seen: set[str] = set()
    for fold_patients in test_patients_seen:
        overlap = seen.intersection(fold_patients)
        if overlap:
            raise RuntimeError(
                "Some patients occur in the test split of multiple folds: "
                + ", ".join(sorted(overlap))
            )
        seen.update(fold_patients)

    if seen != set(unique_patients):
        missing = set(unique_patients).difference(seen)
        raise RuntimeError(
            "Some patients were not assigned to a test fold: "
            + ", ".join(sorted(missing))
        )

    return output
