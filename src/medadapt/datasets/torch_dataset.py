"""PyTorch dataset backed by the audited, deduplicated BreakHis manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - depends on optional training extra
    raise ImportError(
        "PyTorch is required for BreakHisManifestDataset. "
        "Install it with: pip install -e '.[training]'"
    ) from exc

from PIL import Image, UnidentifiedImageError


class BreakHisManifestDataset(Dataset):
    """Load only image records explicitly supplied by a manifest or fold.

    root_dir must be the directory relative to which image_path was written
    by the Phase 1 audit: the histology_slides/breast directory. The dataset
    never recursively scans the source directory.
    """

    REQUIRED_FIELDS = {"image_path", "label", "patient_id"}

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        root_dir: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"BreakHis image root is not a directory: {self.root_dir}"
            )

        self.transform = transform
        self.records = [dict(row) for row in records]
        if not self.records:
            raise ValueError("No image records were supplied to the dataset.")

        self.samples: list[tuple[Path, dict[str, Any]]] = []
        seen_paths: set[str] = set()

        for index, row in enumerate(self.records):
            missing = self.REQUIRED_FIELDS.difference(row)
            if missing:
                raise ValueError(
                    f"Record {index} is missing fields: {', '.join(sorted(missing))}"
                )

            relative = Path(str(row["image_path"]).strip())
            if not str(relative) or relative.is_absolute():
                raise ValueError(
                    f"Record {index} must contain a relative image_path."
                )

            image_path = (self.root_dir / relative).resolve()
            if not image_path.is_relative_to(self.root_dir):
                raise ValueError(
                    f"Image path escapes the dataset root: {row['image_path']}"
                )

            normalized = relative.as_posix()
            if normalized in seen_paths:
                raise ValueError(f"Duplicate image_path in records: {normalized}")
            seen_paths.add(normalized)

            if "parse_status" in row and str(row["parse_status"]) != "ok":
                raise ValueError(
                    f"Image {normalized} has parse_status "
                    f"{row['parse_status']!r}, expected 'ok'."
                )

            try:
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Image {normalized} has an invalid binary label."
                ) from exc
            if label not in (0, 1):
                raise ValueError(
                    f"Image {normalized} has label {label}; expected 0 or 1."
                )
            if not str(row["patient_id"]).strip():
                raise ValueError(f"Image {normalized} has an empty patient_id.")

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image listed in the manifest does not exist: {image_path}"
                )

            row["label"] = label
            self.samples.append((image_path, row))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, row = self.samples[index]
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(
                f"Could not read BreakHis image {image_path}: {exc}"
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": int(row["label"]),
            "patient_id": str(row["patient_id"]),
            "image_path": str(row["image_path"]),
            "subtype_code": str(row.get("subtype_code", "")),
            "magnification": str(row.get("magnification", "")),
        }
