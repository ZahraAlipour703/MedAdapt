"""Create reproducible patient-disjoint BreakHis cross-validation folds."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medadapt.datasets.splits import generate_grouped_folds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/dataset_audit/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/splits/breakhis_5fold.csv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.manifest.exists():
        parser.error(f"Manifest not found: {args.manifest}")
    with args.manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    try:
        split_rows = generate_grouped_folds(rows, args.n_splits, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    print(f"Created {args.n_splits} folds for {len(rows)} images.")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
