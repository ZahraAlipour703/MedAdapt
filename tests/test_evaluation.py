import unittest

from medadapt.evaluation import (
    aggregate_patient_predictions,
    binary_classification_metrics,
)


class EvaluationTests(unittest.TestCase):
    def test_patient_aggregation_averages_image_probabilities(self):
        rows = [
            {"patient_id": "p1", "label": 0, "probability": 0.2},
            {"patient_id": "p1", "label": 0, "probability": 0.4},
            {"patient_id": "p2", "label": 1, "probability": 0.8},
        ]
        result = aggregate_patient_predictions(rows)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["probability"], 0.3)
        self.assertEqual(result[0]["image_count"], 2)

    def test_patient_aggregation_rejects_conflicting_labels(self):
        rows = [
            {"patient_id": "p1", "label": 0, "probability": 0.2},
            {"patient_id": "p1", "label": 1, "probability": 0.4},
        ]
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            aggregate_patient_predictions(rows)

    def test_binary_metrics(self):
        rows = [
            {"label": 0, "probability": 0.1},
            {"label": 0, "probability": 0.3},
            {"label": 1, "probability": 0.7},
            {"label": 1, "probability": 0.9},
        ]
        metrics = binary_classification_metrics(rows)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["average_precision"], 1.0)

    def test_single_class_ranking_metrics_are_none(self):
        rows = [
            {"label": 1, "probability": 0.7},
            {"label": 1, "probability": 0.9},
        ]
        metrics = binary_classification_metrics(rows)
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["average_precision"])
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertIsNone(metrics["specificity"])


if __name__ == "__main__":
    unittest.main()
