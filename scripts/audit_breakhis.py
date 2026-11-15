"""Audit a local BreakHis directory and write a portable CSV manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medadapt.datasets.metadata import find_image_files, find_parent_class, parse_breakhis_filename  # noqa: E402

FIELDS = [
    "image_path", "filename", "biopsy_procedure", "tumor_class_code",
    "label", "class_name", "subtype_code", "subtype_name", "patient_id",
    "magnification", "sequence_number", "parent_class", "parse_status",
    "parse_error", "image_readable", "image_width", "image_height", "sha256",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(root: Path, output_dir: Path, verify_images: bool = True,
                  calculate_hashes: bool = False) -> dict[str, Any]:
    image_paths = find_image_files(root)
    if not image_paths:
        raise RuntimeError(f"No PNG/JPG/JPEG files found beneath {root}")

    manifest = []
    invalid = []
    unreadable = []
    class_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    magnification_counts: Counter[str] = Counter()
    patient_labels: dict[str, set[int]] = defaultdict(set)
    hashes: dict[str, list[str]] = defaultdict(list)
    parent_mismatches = []
    parse_status_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    readable_count = 0

    for path in image_paths:
        rel = path.relative_to(root).as_posix()
        parent_class = find_parent_class(path, root)
        row = {field: "" for field in FIELDS}
        row.update(image_path=rel, filename=path.name, parent_class=parent_class or "",
                   parse_status="unparsed", parse_error="")
        try:
            parsed = parse_breakhis_filename(path.name)
            if parsed is None:
                row["parse_error"] = "Filename does not match expected BreakHis convention."
            else:
                row.update(parsed)
                row["parse_status"] = "ok"
                if parent_class and parent_class != parsed["class_name"]:
                    row["parse_status"] = "parent_class_mismatch"
                    row["parse_error"] = f"Filename says {parsed['class_name']}; directory says {parent_class}."
                    parent_mismatches.append({"image_path": rel, "filename_class": parsed["class_name"],
                                              "parent_class": parent_class})
                else:
                    class_counts[parsed["class_name"]] += 1
                    subtype_counts[parsed["subtype_name"]] += 1
                    magnification_counts[parsed["magnification"]] += 1
                    patient_labels[parsed["patient_id"]].add(parsed["label"])
        except ValueError as exc:
            row["parse_status"] = "metadata_error"
            row["parse_error"] = str(exc)

        parse_status_counts[row["parse_status"]] += 1
        if row["parse_status"] != "ok":
            invalid.append({k: row[k] for k in ("image_path", "filename", "parse_status", "parse_error")})

        if verify_images:
            try:
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                row.update(image_readable=True, image_width=width, image_height=height)
                dimensions[f"{width}x{height}"] += 1
                readable_count += 1
            except Exception as exc:
                row["image_readable"] = False
                unreadable.append({"image_path": rel, "error": f"{type(exc).__name__}: {exc}"})

        if calculate_hashes:
            digest = sha256_file(path)
            row["sha256"] = digest
            hashes[digest].append(rel)
        manifest.append(row)

    duplicates = [
        {"sha256": digest, "image_path": path, "duplicate_group_size": len(paths)}
        for digest, paths in sorted(hashes.items()) if len(paths) > 1 for path in paths
    ]
    conflicts = {patient: sorted(labels) for patient, labels in patient_labels.items() if len(labels) > 1}
    summary = {
        "dataset": "BreakHis", "root": str(root), "total_image_files": len(image_paths),
        "parsed_images": parse_status_counts.get("ok", 0),
        "parse_status_counts": dict(sorted(parse_status_counts.items())),
        "unique_patients": len(patient_labels), "patients_with_conflicting_labels": conflicts,
        "class_counts": dict(sorted(class_counts.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "magnification_counts": dict(sorted(magnification_counts.items())),
        "parent_class_mismatches": parent_mismatches,
        "image_verification_performed": verify_images,
        "readable_images": readable_count if verify_images else None,
        "unreadable_images": len(unreadable) if verify_images else None,
        "image_dimensions": dict(sorted(dimensions.items())),
        "sha256_calculated": calculate_hashes,
        "duplicate_image_files": len(duplicates) if calculate_hashes else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "manifest.csv", manifest, FIELDS)
    write_csv(output_dir / "invalid_images.csv", invalid,
              ["image_path", "filename", "parse_status", "parse_error"])
    write_csv(output_dir / "unreadable_images.csv", unreadable, ["image_path", "error"])
    write_csv(output_dir / "parent_class_mismatches.csv", parent_mismatches,
              ["image_path", "filename_class", "parent_class"])
    write_csv(output_dir / "duplicate_images.csv", duplicates,
              ["sha256", "image_path", "duplicate_group_size"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/BreakHis"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument("--skip-image-verification", action="store_true")
    parser.add_argument("--hash-images", action="store_true")
    args = parser.parse_args()
    try:
        summary = audit_dataset(args.root.resolve(), args.output_dir.resolve(),
                                not args.skip_image_verification, args.hash_images)
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    print(f"\nManifest: {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
