from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medadapt.datasets.splits import generate_grouped_folds


def sample_rows():
    return [
        {"image_path": f"{label}-{patient}/{image}.png", "label": str(label),
         "patient_id": f"{label}-{patient}", "parse_status": "ok"}
        for label in (0, 1) for patient in range(10) for image in range(3)
    ]


class PatientGroupedSplitTests(unittest.TestCase):
    def test_patients_are_disjoint_and_tested_once(self):
        rows = sample_rows()
        result = generate_grouped_folds(rows, n_splits=5, random_state=42)
        tested = []
        for fold in range(1, 6):
            current = [r for r in result if r["fold"] == fold]
            train = {r["patient_id"] for r in current if r["split"] == "train"}
            test = {r["patient_id"] for r in current if r["split"] == "test"}
            self.assertTrue(train.isdisjoint(test))
            tested.extend(test)
        self.assertEqual(len(tested), 20)
        self.assertEqual(len(set(tested)), 20)

    def test_conflicting_patient_labels_are_rejected(self):
        rows = sample_rows()
        rows.append({"image_path": "x.png", "label": "1", "patient_id": "0-0", "parse_status": "ok"})
        with self.assertRaises(ValueError):
            generate_grouped_folds(rows)

    def test_invalid_metadata_is_rejected(self):
        rows = sample_rows()
        rows[0]["parse_status"] = "unparsed"
        with self.assertRaises(ValueError):
            generate_grouped_folds(rows)


if __name__ == "__main__":
    unittest.main()
