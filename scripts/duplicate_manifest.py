
"""Create reproducible deduplicated BreakHis manifests from pixel hashes."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_manifests(manifest_path: Path, duplicates_path: Path, output_dir: Path):
    manifest = read_csv(manifest_path)
    duplicate_rows = read_csv(duplicates_path)

    if not manifest:
        raise ValueError("The source manifest is empty.")

    by_path = {row["image_path"]: row for row in manifest}
    if len(by_path) != len(manifest):
        raise ValueError("Duplicate image_path entries in source manifest.")

    groups = defaultdict(list)

    for row in duplicate_rows:
        path = row["image_path"]
        digest = row["sha256_rgb"]

        if path not in by_path:
            raise ValueError(f"Duplicate report path missing from manifest: {path}")

        groups[digest].append(path)

    grouped_paths = set()
    for digest, paths in groups.items():
        if len(paths) != len(set(paths)):
            raise ValueError(f"Repeated path in duplicate group {digest}")
        grouped_paths.update(paths)

    if len(grouped_paths) != sum(len(v) for v in groups.values()):
        raise ValueError("An image occurs in more than one duplicate group.")

    records = []
    provenance = []

    # Every original image gets a unique grouping key unless its pixel
    # hash identifies it as part of a duplicate group.
    for row in manifest:
        path = row["image_path"]
        records.append(row.copy())

    representatives = {}
    for digest, paths in groups.items():
        ordered = sorted(paths)
        representative = ordered[0]
        representatives[digest] = representative

        members = [by_path[p] for p in ordered]
        labels = {str(r["label"]) for r in members}
        patients = {str(r["patient_id"]) for r in members}
        subtypes = {str(r["subtype_code"]) for r in members}

        if len(labels) != 1:
            raise ValueError(
                f"Pixel duplicate group {digest} crosses binary labels."
            )
        if len(patients) != 1:
            raise ValueError(
                f"Pixel duplicate group {digest} crosses patient IDs."
            )

        for path in ordered:
            provenance.append({
                "sha256_rgb": digest,
                "patient_id": by_path[path]["patient_id"],
                "representative_path": representative,
                "image_path": path,
                "is_representative": path == representative,
                "subtype_codes": ";".join(sorted(subtypes)),
                "subtype_ambiguous": len(subtypes) > 1,
                "binary_label": next(iter(labels)),
            })

    # Retain one representative per exact-pixel duplicate group.
    # Keep all non-duplicate images.
    deduplicated = []
    ambiguous_subtype_paths = set()

    for digest, paths in groups.items():
        if len({by_path[p]["subtype_code"] for p in paths}) > 1:
            ambiguous_subtype_paths.add(representatives[digest])

    duplicate_representatives = set(representatives.values())

    for row in records:
        path = row["image_path"]

        if path in grouped_paths and path not in duplicate_representatives:
            continue

        if path in ambiguous_subtype_paths:
            row["subtype_code"] = "AMBIGUOUS"
            row["subtype_name"] = "ambiguous_duplicate"

        row["pixel_duplicate_group"] = ""
        for digest, representative in representatives.items():
            if path == representative:
                row["pixel_duplicate_group"] = digest

        deduplicated.append(row)

    # Strict subtype manifest excludes representatives whose subtype
    # cannot be determined from the duplicate group.
    subtype_manifest = [
        row for row in deduplicated
        if row["subtype_code"] != "AMBIGUOUS"
    ]

    fields = list(manifest[0])
    if "pixel_duplicate_group" not in fields:
        fields.append("pixel_duplicate_group")

    write_csv(
        output_dir / "breakhis_binary_deduplicated.csv",
        deduplicated,
        fields,
    )
    write_csv(
        output_dir / "breakhis_subtype_unambiguous.csv",
        subtype_manifest,
        fields,
    )
    write_csv(
        output_dir / "duplicate_provenance.csv",
        provenance,
        [
            "sha256_rgb", "patient_id", "representative_path",
            "image_path", "is_representative", "subtype_codes",
            "subtype_ambiguous", "binary_label",
        ],
    )

    summary = {
        "original_images": len(manifest),
        "duplicate_groups": len(groups),
        "redundant_images_removed": len(grouped_paths) - len(groups),
        "binary_deduplicated_images": len(deduplicated),
        "ambiguous_subtype_groups": len(ambiguous_subtype_paths),
        "subtype_unambiguous_images": len(subtype_manifest),
        "patient_groups_binary": len({
            r["patient_id"] for r in deduplicated
        }),
        "patient_groups_subtype": len({
            r["patient_id"] for r in subtype_manifest
        }),
    }

    import json
    (output_dir / "deduplication_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("outputs/dataset_audit/manifest.csv")
    )
    parser.add_argument(
        "--duplicates", type=Path,
        default=Path("outputs/dataset_audit/pixel_duplicates.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/deduplicated")
    )
    args = parser.parse_args()

    summary = build_manifests(
        args.manifest, args.duplicates, args.output_dir
    )
    print("=== DEDUPLICATION SUMMARY ===")
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
