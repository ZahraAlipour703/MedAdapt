from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medadapt.datasets.metadata import parse_breakhis_filename


class BreakHisMetadataTests(unittest.TestCase):
    def test_standard_filename(self):
        row = parse_breakhis_filename("SOB_B_TA-14-4659-40-001.png")
        self.assertEqual(row["label"], 0)
        self.assertEqual(row["subtype_name"], "tubular_adenoma")
        self.assertEqual(row["patient_id"], "14-4659")
        self.assertEqual(row["magnification"], "40X")

    def test_alphanumeric_patient_suffix(self):
        row = parse_breakhis_filename("SOB_M_DC-14-22549AB-400-012.png")
        self.assertEqual(row["patient_id"], "14-22549AB")
        self.assertEqual(row["label"], 1)

    def test_unrecognized_filename(self):
        self.assertIsNone(parse_breakhis_filename("image_001.png"))

    def test_class_subtype_conflict(self):
        with self.assertRaises(ValueError):
            parse_breakhis_filename("SOB_B_DC-14-4659-40-001.png")


if __name__ == "__main__":
    unittest.main()
