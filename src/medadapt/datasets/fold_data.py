"""Patient-disjoint outer/inner split helpers and PyTorch DataLoader builders."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from sklearn.model_selection import StratifiedGroupKFold


def _validate_records(records: list[dict[str, Any]], context: str) -> None:
    if not records:
        raise ValueError(f"No records found for {context}.")

    required = {"image_path", "label", "patient_id"}
    missing = required.difference(records[0])
    if missing:
        raise ValueError(
            f"{context} is missing columns: {', '.join(sorted(missing))}"
        )

    seen_paths: set[str] = set()
    patient_labels: dict[str, int] = {}
    for row in records:
        path = str(row["image_path"]).strip()
        patient = str(row["patient_id"]).strip()
        try:
            label = int(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid label for image {path!r}.") from exc

        if not path or not patient or label not in (0, 1):
            raise ValueError(f"Invalid image_path, patient_id, or label in {context}.")
        if path in seen_paths:
            raise ValueError(f"Duplicate image_path in {context}: {path}")
        seen_paths.add(path)

        previous = patient_labels.setdefault(patient, label)
        if previous != label:
            raise ValueError(
                f"Patient {patient!r} has conflicting binary labels in {context}."
            )


def load_outer_fold_rows(
    fold_csv_path: str | Path,
    fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and validate the train/test records for one outer fold."""
    if fold < 1:
        raise ValueError("Fold numbers start at 1.")

    path = Path(fold_csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fold CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "label", "patient_id", "fold", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Fold CSV is missing columns: {', '.join(sorted(missing))}"
            )
        selected = [
            dict(row) for row in reader
            if str(row["fold"]).strip() == str(fold)
        ]

    if not selected:
        raise ValueError(f"Fold {fold} was not found in {path}.")

    split_names = {str(row["split"]).strip() for row in selected}
    if split_names != {"train", "test"}:
        raise ValueError(
            f"Fold {fold} must contain train and test records; found {split_names}."
        )

    train_rows = [row for row in selected if row["split"] == "train"]
    test_rows = [row for row in selected if row["split"] == "test"]
    _validate_records(train_rows, f"fold {fold} training partition")
    _validate_records(test_rows, f"fold {fold} test partition")

    train_patients = {str(row["patient_id"]) for row in train_rows}
    test_patients = {str(row["patient_id"]) for row in test_rows}
    if not train_patients.isdisjoint(test_patients):
        overlap = sorted(train_patients.intersection(test_patients))
        raise ValueError(
            f"Patient leakage in outer fold {fold}: {', '.join(overlap)}"
        )

    train_paths = {str(row["image_path"]) for row in train_rows}
    test_paths = {str(row["image_path"]) for row in test_rows}
    if not train_paths.isdisjoint(test_paths):
        raise ValueError(f"Image leakage in outer fold {fold}.")

    return train_rows, test_rows


def split_training_validation(
    training_rows: Iterable[dict[str, Any]],
    n_splits: int = 5,
    validation_fold: int = 0,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a patient-disjoint inner train/validation split.

    validation_fold is zero-based. The outer test partition must not be passed
    to this function; it is kept separate by load_outer_fold_rows.
    """
    records = [dict(row) for row in training_rows]
    _validate_records(records, "inner split input")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if not 0 <= validation_fold < n_splits:
        raise ValueError("validation_fold must be in [0, n_splits).")

    patient_labels: dict[str, int] = {}
    for row in records:
        patient = str(row["patient_id"])
        label = int(row["label"])
        patient_labels.setdefault(patient, label)

    class_patient_counts = {
        label: sum(value == label for value in patient_labels.values())
        for label in (0, 1)
    }
    if min(class_patient_counts.values()) < n_splits:
        raise ValueError(
            f"Not enough patients per class for {n_splits} inner folds: "
            f"{class_patient_counts}."
        )

    labels = [int(row["label"]) for row in records]
    groups = [str(row["patient_id"]) for row in records]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    partitions = list(
        splitter.split(X=[0] * len(records), y=labels, groups=groups)
    )
    train_indices, validation_indices = partitions[validation_fold]

    inner_train = [records[int(index)] for index in train_indices]
    validation = [records[int(index)] for index in validation_indices]

    _validate_records(inner_train, "inner training partition")
    _validate_records(validation, "inner validation partition")

    train_patients = {str(row["patient_id"]) for row in inner_train}
    validation_patients = {str(row["patient_id"]) for row in validation}
    if not train_patients.isdisjoint(validation_patients):
        raise RuntimeError("Patient leakage detected in inner train/validation split.")

    if {int(row["label"]) for row in inner_train} != {0, 1}:
        raise ValueError("Inner training partition does not contain both classes.")
    if {int(row["label"]) for row in validation} != {0, 1}:
        raise ValueError("Inner validation partition does not contain both classes.")

    return inner_train, validation



def patient_class_balanced_weights(
    records: Iterable[dict[str, Any]],
) -> list[float]:
    """Return per-image weights giving equal mass to each patient and class.

    The weights sum to one. Each binary class receives total weight 0.5,
    and each patient within a class receives equal total weight, regardless
    of how many images that patient contributes.
    """
    rows = [dict(row) for row in records]
    _validate_records(rows, "patient-balanced sampler input")

    patient_image_counts: dict[str, int] = {}
    patient_labels: dict[str, int] = {}
    for row in rows:
        patient = str(row["patient_id"])
        label = int(row["label"])
        patient_image_counts[patient] = patient_image_counts.get(patient, 0) + 1
        patient_labels[patient] = label

    patients_per_class = {
        label: sum(value == label for value in patient_labels.values())
        for label in (0, 1)
    }
    if any(count == 0 for count in patients_per_class.values()):
        raise ValueError(
            "Patient-class-balanced sampling requires patients from both classes."
        )

    weights = []
    for row in rows:
        patient = str(row["patient_id"])
        label = int(row["label"])
        weights.append(
            1.0 / (
                2.0
                * patients_per_class[label]
                * patient_image_counts[patient]
            )
        )

    return weights



def build_fold_dataloaders(
    root_dir: str | Path,
    fold_csv_path: str | Path,
    fold: int,
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 224,
    validation_splits: int = 5,
    validation_fold: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
    sampling_strategy: str = "patient_class_balanced",
) -> dict[str, Any]:
    """Build train/validation/test DataLoaders for one outer fold."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if sampling_strategy not in {"patient_class_balanced", "image_uniform"}:
        raise ValueError(
            "sampling_strategy must be 'patient_class_balanced' or 'image_uniform'."
        )

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise ImportError(
            "PyTorch is required to build DataLoaders. "
            "Install it with: pip install -e '.[training]'"
        ) from exc

    from .torch_dataset import BreakHisManifestDataset
    from .transforms import build_transforms

    outer_train, outer_test = load_outer_fold_rows(fold_csv_path, fold)
    train_rows, validation_rows = split_training_validation(
        outer_train,
        n_splits=validation_splits,
        validation_fold=validation_fold,
        random_state=seed,
    )

    train_dataset = BreakHisManifestDataset(
        train_rows, root_dir, transform=build_transforms(image_size, train=True)
    )
    validation_dataset = BreakHisManifestDataset(
        validation_rows, root_dir, transform=build_transforms(image_size, train=False)
    )
    test_dataset = BreakHisManifestDataset(
        outer_test, root_dir, transform=build_transforms(image_size, train=False)
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    if sampling_strategy == "patient_class_balanced":
        sample_weights = patient_class_balanced_weights(train_rows)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset, sampler=sampler, shuffle=False, **common
        )
    else:
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=generator, **common
        )

    return {
        "sampling_strategy": sampling_strategy,
        "train": train_loader,
        "validation": DataLoader(
            validation_dataset, shuffle=False, **common
        ),
        "test": DataLoader(
            test_dataset, shuffle=False, **common
        ),
        "records": {
            "train": train_rows,
            "validation": validation_rows,
            "test": outer_test,
        },
    }
