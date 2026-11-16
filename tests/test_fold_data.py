import csv
from pathlib import Path
import tempfile
import unittest

from medadapt.datasets.fold_data import (
    load_outer_fold_rows,
    split_training_validation,
    patient_class_balanced_weights,
)


def make_rows():
    rows = []
    for label in (0, 1):
        for patient_index in range(10):
            patient = f"{label}-{patient_index}"
            for image_index in range(2):
                rows.append({
                    "image_path": f"{patient}/{image_index}.png",
                    "label": str(label),
                    "patient_id": patient,
                })
    return rows


class FoldDataTests(unittest.TestCase):

    def test_patient_class_balanced_weights_equalize_patient_and_class_mass(self):
        rows = make_rows()
        # Make one patient contribute substantially more images.
        rows.extend([
            {
                "image_path": f"0-0/extra-{i}.png",
                "label": "0",
                "patient_id": "0-0",
            }
            for i in range(8)
        ])

        weights = patient_class_balanced_weights(rows)
        self.assertAlmostEqual(sum(weights), 1.0)

        patient_mass = {}
        class_mass = {}
        for row, weight in zip(rows, weights):
            patient = row["patient_id"]
            label = int(row["label"])
            patient_mass[patient] = patient_mass.get(patient, 0.0) + weight
            class_mass[label] = class_mass.get(label, 0.0) + weight

        for label in (0, 1):
            masses = [
                mass for patient, mass in patient_mass.items()
                if int(patient.split("-")[0]) == label
            ]
            self.assertTrue(all(abs(mass - masses[0]) < 1e-12 for mass in masses))
            self.assertAlmostEqual(class_mass[label], 0.5)

    def test_patient_class_balanced_weights_require_both_classes(self):
        rows = [row for row in make_rows() if row["label"] == "1"]
        with self.assertRaisesRegex(ValueError, "both classes"):
            patient_class_balanced_weights(rows)

    def test_inner_split_is_patient_disjoint_and_complete(self):
        rows = make_rows()
        train, validation = split_training_validation(
            rows, n_splits=5, validation_fold=0, random_state=17
        )
        train_patients = {row["patient_id"] for row in train}
        validation_patients = {row["patient_id"] for row in validation}

        self.assertTrue(train_patients.isdisjoint(validation_patients))
        self.assertEqual(len(train) + len(validation), len(rows))
        self.assertEqual(
            {row["image_path"] for row in train + validation},
            {row["image_path"] for row in rows},
        )
        self.assertEqual({int(row["label"]) for row in train}, {0, 1})
        self.assertEqual({int(row["label"]) for row in validation}, {0, 1})

    def test_inner_split_is_reproducible(self):
        rows = make_rows()
        first = split_training_validation(rows, random_state=17)
        second = split_training_validation(rows, random_state=17)
        self.assertEqual(
            [row["image_path"] for row in first[0]],
            [row["image_path"] for row in second[0]],
        )
        self.assertEqual(
            [row["image_path"] for row in first[1]],
            [row["image_path"] for row in second[1]],
        )

    def test_outer_fold_reader_checks_patient_leakage(self):
        rows = make_rows()
        for row in rows:
            row["fold"] = "1"
            row["split"] = (
                "test" if row["patient_id"] in {"0-0", "1-0"} else "train"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "folds.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(rows[0]) + ["fold", "split"]
                )
                writer.writeheader()
                writer.writerows(rows)

            train, test = load_outer_fold_rows(path, 1)
            self.assertEqual(len(train) + len(test), len(rows))
            self.assertTrue(
                {r["patient_id"] for r in train}.isdisjoint(
                    {r["patient_id"] for r in test}
                )
            )

    def test_outer_fold_reader_rejects_patient_leakage(self):
        rows = make_rows()
        for row in rows:
            row["fold"] = "1"
            row["split"] = "test" if row["patient_id"] == "0-0" else "train"
        rows.append({
            **rows[0],
            "image_path": "leakage/copy.png",
            "fold": "1",
            "split": "train",
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "folds.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "Patient leakage"):
                load_outer_fold_rows(path, 1)


if __name__ == "__main__":
    unittest.main()
