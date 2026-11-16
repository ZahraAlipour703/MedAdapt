"""Reproducible ResNet training and inference utilities."""
from __future__ import annotations

import random
import time
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, loader, device, max_batches: int | None = None):
    """Return per-image predictions in loader order."""
    import torch

    model.eval()
    output = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].tolist()
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            for path, patient, label, probability in zip(
                batch["image_path"], batch["patient_id"], labels, probabilities
            ):
                output.append({
                    "image_path": str(path),
                    "patient_id": str(patient),
                    "label": int(label),
                    "probability": float(probability),
                })
    return output, time.perf_counter() - started


def train_one_epoch(model, loader, optimizer, criterion, device,
                    max_batches: int | None = None) -> dict[str, float | int]:
    import torch

    model.train()
    total_loss = 0.0
    total_samples = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss encountered.")
        loss.backward()
        optimizer.step()
        size = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * size
        total_samples += size
    if not total_samples:
        raise RuntimeError("Training loader produced no batches.")
    return {
        "loss": total_loss / total_samples,
        "samples": total_samples,
        "seconds": time.perf_counter() - started,
    }
