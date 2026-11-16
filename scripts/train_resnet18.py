"""Train/evaluate ResNet-18 on one patient-grouped BreakHis outer fold."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn

from medadapt.datasets.fold_data import build_fold_dataloaders
from medadapt.evaluation import (
    aggregate_patient_predictions,
    binary_classification_metrics,
)
from medadapt.models import build_resnet18
from medadapt.models.resnet18 import count_parameters
from medadapt.training import predict, seed_everything, train_one_epoch


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--fold-csv", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output", default="outputs/experiments")
    parser.add_argument("--sampling", choices=[
        "patient_class_balanced", "image_uniform"
    ], default="patient_class_balanced")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Debug/smoke-test limit; do not use for reported results.")
    parser.add_argument("--max-eval-batches", type=int, default=None,
                        help="Debug/smoke-test limit; do not use for reported results.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive.")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("max-train-batches must be positive.")
    if args.max_eval_batches is not None and args.max_eval_batches < 1:
        parser.error("max-eval-batches must be positive.")

    seed_everything(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable.")

    loaders = build_fold_dataloaders(
        root_dir=args.root, fold_csv_path=args.fold_csv, fold=args.fold,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, seed=args.seed,
        sampling_strategy=args.sampling,
    )
    model = build_resnet18(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    smoke_test = (
        args.max_train_batches is not None or args.max_eval_batches is not None
    )
    run_name = f"resnet18_fold{args.fold}_{args.sampling}_seed{args.seed}"
    if smoke_test:
        run_name += "_smoke"
    run_dir = Path(args.output) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pt"

    history = []
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device,
            max_batches=args.max_train_batches,
        )
        val_predictions, val_seconds = predict(
            model, loaders["validation"], device,
            max_batches=args.max_eval_batches,
        )
        val_patient = aggregate_patient_predictions(val_predictions)
        val_metrics = binary_classification_metrics(val_patient)
        score = float(val_metrics["log_loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_samples": train_stats["samples"],
            "train_seconds": train_stats["seconds"],
            "validation_seconds": val_seconds,
            "validation_patient_log_loss": score,
            "validation_patient_roc_auc": val_metrics["roc_auc"],
            "validation_patient_average_precision": val_metrics["average_precision"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if score < best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_patient_log_loss": score,
                "args": vars(args),
            }, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    elapsed = time.perf_counter() - started
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    inference_seconds = None
    image_predictions = None
    patient_predictions = None
    image_metrics = None
    patient_metrics = None

    if smoke_test:
        print(
            "SMOKE TEST: outer test inference skipped; "
            "limited training/validation runs are not reportable.",
            flush=True,
        )
    else:
        image_predictions, inference_seconds = predict(
            model, loaders["test"], device
        )
        patient_predictions = aggregate_patient_predictions(image_predictions)
        image_metrics = binary_classification_metrics(image_predictions)
        patient_metrics = binary_classification_metrics(patient_predictions)
        write_csv(run_dir / "test_image_predictions.csv", image_predictions)
        write_csv(run_dir / "test_patient_predictions.csv", patient_predictions)

    write_csv(run_dir / "history.csv", history)
    results = {
        "is_smoke_test": smoke_test,
        "experiment": "resnet18_transfer_learning",
        "fold": args.fold,
        "sampling_strategy": args.sampling,
        "seed": args.seed,
        "device": str(device),
        "pretrained_imagenet": args.pretrained,
        "best_epoch": best_epoch,
        "best_validation_patient_log_loss": best_score,
        "epochs_completed": len(history),
        "training_and_validation_seconds": elapsed,
        "test_inference_seconds": inference_seconds,
        "test_images_per_second": (
            len(image_predictions) / inference_seconds
            if image_predictions is not None and inference_seconds else None
        ),
        "parameters": count_parameters(model),
        "image_level_test_metrics": image_metrics,
        "patient_level_test_metrics": patient_metrics,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "smoke_test_limits": {
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
        },
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\n=== SMOKE TEST SUMMARY ===" if smoke_test else "\n=== FINAL TEST RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"Artifacts: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
