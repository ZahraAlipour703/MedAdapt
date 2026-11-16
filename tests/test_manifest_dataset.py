import importlib.util
from pathlib import Path
import tempfile
import unittest

TORCH_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchvision") is not None
)

if TORCH_AVAILABLE:
    import torch
    from PIL import Image
    from medadapt.datasets.torch_dataset import BreakHisManifestDataset
    from medadapt.datasets.transforms import build_transforms


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch and TorchVision training extras are optional.")
class ManifestDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.relative_path = Path("benign/SOB/adenosis/sample.png")
        image_path = self.root / self.relative_path
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (80, 48), color=(140, 90, 120)).save(image_path)
        self.row = {
            "image_path": self.relative_path.as_posix(),
            "label": "0",
            "patient_id": "patient-1",
            "parse_status": "ok",
            "subtype_code": "A",
            "magnification": "40X",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_loads_rgb_tensor_and_metadata(self):
        dataset = BreakHisManifestDataset(
            [self.row],
            self.root,
            transform=build_transforms(image_size=32, train=False),
        )
        sample = dataset[0]

        self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))
        self.assertEqual(sample["label"], 0)
        self.assertEqual(sample["patient_id"], "patient-1")
        self.assertEqual(sample["subtype_code"], "A")
        self.assertTrue(torch.isfinite(sample["image"]).all())

    def test_training_and_evaluation_transforms_return_same_shape(self):
        image_path = self.root / self.relative_path
        with Image.open(image_path) as image:
            train_image = build_transforms(32, train=True)(image.convert("RGB"))
            eval_image = build_transforms(32, train=False)(image.convert("RGB"))

        self.assertEqual(tuple(train_image.shape), (3, 32, 32))
        self.assertEqual(tuple(eval_image.shape), (3, 32, 32))

    def test_dataset_rejects_path_traversal(self):
        row = {**self.row, "image_path": "../outside.png"}
        with self.assertRaisesRegex(ValueError, "escapes"):
            BreakHisManifestDataset([row], self.root)

    def test_dataset_rejects_duplicate_paths(self):
        with self.assertRaisesRegex(ValueError, "Duplicate image_path"):
            BreakHisManifestDataset([self.row, self.row], self.root)

    def test_dataset_rejects_missing_image(self):
        row = {**self.row, "image_path": "missing.png"}
        with self.assertRaises(FileNotFoundError):
            BreakHisManifestDataset([row], self.root)


if __name__ == "__main__":
    unittest.main()
