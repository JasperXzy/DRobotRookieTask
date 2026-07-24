#!/usr/bin/env python3
"""Train the Task2 ResNet18 two-output baseline or three-output model."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common import (
    load_yaml,
    resolve_repo_path,
    set_seed,
    sha256_file,
    write_json,
)
from dataset import TrackDataset
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "Task2/attachments/track_detection_resnet18/"
            "configs/train_resnet18_xyv.yaml"
        ),
    )
    parser.add_argument(
        "--manifest",
        default="Task2/data/dataset_manifest.csv",
    )
    parser.add_argument(
        "--run-dir",
        help="Run output directory; defaults to Task2/runs/<UTC timestamp>.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train the positive-only [x_norm,y_norm] teaching baseline.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download/use ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def compute_loss(
    outputs: torch.Tensor,
    xy: torch.Tensor,
    has_track: torch.Tensor,
    confidence_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    positive = has_track >= 0.5
    positive_count = int(positive.sum().item())
    if positive_count:
        coordinate = functional.smooth_l1_loss(
            outputs[positive, :2],
            xy[positive],
        )
    else:
        coordinate = outputs[:, :2].sum() * 0.0

    if outputs.shape[1] == 3:
        confidence = functional.binary_cross_entropy_with_logits(
            outputs[:, 2],
            has_track,
        )
        total = coordinate + confidence_weight * confidence
    else:
        confidence = outputs.sum() * 0.0
        total = coordinate
    return total, coordinate, confidence, positive_count


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    confidence_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "coordinate": 0.0, "confidence": 0.0}
    sample_count = 0
    positive_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        xy = batch["xy"].to(device)
        has_track = batch["has_track"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            total, coordinate, confidence, batch_positive = compute_loss(
                outputs,
                xy,
                has_track,
                confidence_weight,
            )
            if training:
                total.backward()
                optimizer.step()

        batch_size = images.shape[0]
        totals["total"] += float(total.detach()) * batch_size
        totals["coordinate"] += float(coordinate.detach()) * batch_size
        totals["confidence"] += float(confidence.detach()) * batch_size
        sample_count += batch_size
        positive_count += batch_positive

    if not sample_count:
        raise RuntimeError("DataLoader produced no batches")
    return {
        "total_loss": totals["total"] / sample_count,
        "coordinate_loss": totals["coordinate"] / sample_count,
        "confidence_loss": totals["confidence"] / sample_count,
        "sample_count": float(sample_count),
        "positive_count": float(positive_count),
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(args.config)
    manifest_path = resolve_repo_path(args.manifest)
    config = load_yaml(config_path)
    training_config = config["training"]
    output_size = 2 if args.baseline else int(config["model"]["output_size"])
    seed = int(training_config["seed"])
    set_seed(seed)
    device = choose_device(args.device)

    if args.run_dir:
        run_dir = resolve_repo_path(args.run_dir)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = "xy" if output_size == 2 else "xyv"
        run_dir = resolve_repo_path(f"Task2/runs/{run_id}_{suffix}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "tensorboard").mkdir()
    shutil.copy2(config_path, run_dir / "config.yaml")

    positive_only = output_size == 2
    train_dataset = TrackDataset(
        manifest_path,
        config,
        split="train",
        training=True,
        positive_only=positive_only,
    )
    val_dataset = TrackDataset(
        manifest_path,
        config,
        split="val",
        training=False,
        positive_only=positive_only,
    )
    loader_options = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(training_config["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = build_model(
        output_size=output_size,
        pretrained=bool(config["model"]["pretrained"])
        and not args.no_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    confidence_weight = float(config["loss"]["confidence_weight"])
    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    writer = SummaryWriter(run_dir / "tensorboard")
    metrics_path = run_dir / "metrics.csv"
    metric_fields = [
        "epoch",
        "train_total_loss",
        "train_coordinate_loss",
        "train_confidence_loss",
        "val_total_loss",
        "val_coordinate_loss",
        "val_confidence_loss",
    ]

    metadata: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "config": str(config_path),
        "seed": seed,
        "output_size": output_size,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    write_json(run_dir / "run_metadata.json", metadata)

    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
        metrics_writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                confidence_weight,
                optimizer,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                confidence_weight,
                optimizer=None,
            )
            row = {
                "epoch": epoch,
                **{
                    f"train_{name}": value
                    for name, value in train_metrics.items()
                    if name.endswith("_loss")
                },
                **{
                    f"val_{name}": value
                    for name, value in val_metrics.items()
                    if name.endswith("_loss")
                },
            }
            metrics_writer.writerow(row)
            metrics_file.flush()
            for name, value in row.items():
                if name != "epoch":
                    writer.add_scalar(name, value, epoch)
            print(json.dumps(row, ensure_ascii=False))

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "output_size": output_size,
                "epoch": epoch,
                "config": config,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, run_dir / "last_model.pth")
            if val_metrics["total_loss"] < best_loss:
                best_loss = val_metrics["total_loss"]
                best_epoch = epoch
                stale_epochs = 0
                torch.save(checkpoint, run_dir / "best_model.pth")
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    writer.close()
    metadata.update(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best_epoch,
            "best_val_total_loss": best_loss,
        }
    )
    write_json(run_dir / "run_metadata.json", metadata)
    print(f"Best checkpoint: {run_dir / 'best_model.pth'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
