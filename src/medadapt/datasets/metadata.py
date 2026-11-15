"""Filename parsing utilities for the BreakHis dataset.

BreakHis filenames encode biopsy procedure, binary diagnosis, histological
subtype, patient/slide identifier, magnification, and image sequence.

Example: SOB_B_TA-14-4659-40-001.png
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SUBTYPES = {
    "A": ("adenosis", "benign"),
    "F": ("fibroadenoma", "benign"),
    "PT": ("phyllodes_tumor", "benign"),
    "TA": ("tubular_adenoma", "benign"),
    "DC": ("ductal_carcinoma", "malignant"),
    "LC": ("lobular_carcinoma", "malignant"),
    "MC": ("mucinous_carcinoma", "malignant"),
    "PC": ("papillary_carcinoma", "malignant"),
}

_FILENAME_RE = re.compile(
    r"^(?P<procedure>[A-Z]+)_(?P<class_code>[BM])_"
    r"(?P<subtype>A|F|PT|TA|DC|LC|MC|PC)-"
    r"(?P<patient_id>\d+-\d+[A-Z]*)-"
    r"(?P<magnification>40|100|200|400)-"
    r"(?P<sequence>\d+)\.(?P<extension>png|jpe?g)$",
    flags=re.IGNORECASE,
)


def parse_breakhis_filename(filename: str | Path) -> dict[str, Any] | None:
    """Parse a BreakHis image filename.

    Returns None when the filename does not follow the expected pattern.
    Raises ValueError when the encoded diagnosis conflicts with its subtype.
    The patient identifier is retained in full (for example 14-22549AB).
    """
    name = Path(filename).name
    match = _FILENAME_RE.fullmatch(name)
    if match is None:
        return None

    data = match.groupdict()
    procedure = data["procedure"].upper()
    class_code = data["class_code"].upper()
    subtype_code = data["subtype"].upper()
    subtype_name, expected_class = SUBTYPES[subtype_code]
    encoded_class = "benign" if class_code == "B" else "malignant"

    if encoded_class != expected_class:
        raise ValueError(
            f"{name}: class code {class_code} conflicts with "
            f"subtype {subtype_code} ({expected_class})"
        )

    return {
        "biopsy_procedure": procedure,
        "tumor_class_code": class_code,
        "label": 0 if encoded_class == "benign" else 1,
        "class_name": encoded_class,
        "subtype_code": subtype_code,
        "subtype_name": subtype_name,
        "patient_id": data["patient_id"].upper(),
        "magnification": f'{data["magnification"]}X',
        "sequence_number": int(data["sequence"]),
        "file_extension": data["extension"].lower(),
    }


def find_image_files(root: str | Path) -> list[Path]:
    """Return supported image files below root in stable path order."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root_path}")

    extensions = {".png", ".jpg", ".jpeg"}
    return sorted(
        (
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: path.as_posix().lower(),
    )


def find_parent_class(path: str | Path, root: str | Path) -> str | None:
    """Find a benign/malignant directory between path and root."""
    path = Path(path)
    root = Path(root)
    try:
        relative_parent_parts = path.parent.relative_to(root).parts
    except ValueError:
        relative_parent_parts = path.parent.parts

    for part in reversed(relative_parent_parts):
        normalized = part.strip().lower()
        if normalized in {"benign", "malignant"}:
            return normalized
    return None
