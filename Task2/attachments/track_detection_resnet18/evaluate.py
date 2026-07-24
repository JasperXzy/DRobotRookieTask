#!/usr/bin/env python3
"""Evaluate a Task2 checkpoint on a frozen manifest split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import load_yaml, resolve_repo_path, write_json
from dataset import TrackDataset
from metrics import compute_metrics, norm_to_pixels, sigmoid
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-dir",
        default="Task2/output/evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--render-largest", type=int, default=30)
    return parser.parse_args()


def render_largest_errors(
    rows: list[dict[str, str | float]],
    output_dir: Path,
    limit: int,
) -> None:
    target_dir = output_dir / "largest_errors"
    target_dir.mkdir(parents=True, exist_ok=True)
    positive_rows = [
        row for row in rows if isinstance(row["distance_error_px"], float)
    ]
    positive_rows.sort(
        key=lambda row: float(row["distance_error_px"]),
        reverse=True,
    )
    for row in positive_rows[:limit]:
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        predicted = (round(float(row["pred_x_px"])), round(float(row["pred_y_px"])))
        target = (round(float(row["target_x_px"])), round(float(row["target_y_px"])))
        cv2.circle(image, target, 6, (0, 255, 0), -1)
        cv2.circle(image, predicted, 6, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"error={float(row['distance_error_px']):.1f}px",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(target_dir / f"{row['sample_id']}.jpg"), image)


def grouped_metrics(
    values: list[str],
    predictions: np.ndarray,
    targets: np.ndarray,
    has_track: np.ndarray,
    threshold: float,
) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    value_array = np.asarray(values, dtype=object)
    for value in sorted(set(values)):
        if not value:
            continue
        mask = value_array == value
        reports[value] = compute_metrics(
            predictions[mask],
            targets[mask],
            has_track[mask],
            threshold,
        )
    return reports


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    config = load_yaml(args.config)
    model, metadata = load_checkpoint(
        resolve_repo_path(args.checkpoint),
        device=device,
    )
    output_size = int(metadata.get("output_size", 3))
    dataset = TrackDataset(
        resolve_repo_path(args.manifest),
        config,
        split=args.split,
        training=False,
        positive_only=output_size == 2,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    track_batches: list[np.ndarray] = []
    sample_ids: list[str] = []
    image_paths: list[str] = []
    video_ids: list[str] = []
    directions: list[str] = []
    lighting_values: list[str] = []
    tasks: list[str] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device)).cpu().numpy()
            prediction_batches.append(outputs)
            target_batches.append(batch["xy"].numpy())
            track_batches.append(batch["has_track"].numpy())
            sample_ids.extend(batch["sample_id"])
            image_paths.extend(batch["image_path"])
            video_ids.extend(batch["video_id"])
            directions.extend(batch["direction"])
            lighting_values.extend(batch["lighting"])
            tasks.extend(batch["task"])

    predictions = np.concatenate(prediction_batches)
    targets = np.concatenate(target_batches)
    has_track = np.concatenate(track_batches).astype(np.int64)
    threshold = float(config["evaluation"]["confidence_threshold"])
    report = compute_metrics(predictions, targets, has_track, threshold)
    report.update(
        {
            "checkpoint": str(resolve_repo_path(args.checkpoint)),
            "manifest": str(resolve_repo_path(args.manifest)),
            "split": args.split,
            "by_video_id": grouped_metrics(
                video_ids,
                predictions,
                targets,
                has_track,
                threshold,
            ),
            "by_direction": grouped_metrics(
                directions,
                predictions,
                targets,
                has_track,
                threshold,
            ),
            "by_lighting": grouped_metrics(
                lighting_values,
                predictions,
                targets,
                has_track,
                threshold,
            ),
            "by_task": grouped_metrics(
                tasks,
                predictions,
                targets,
                has_track,
                threshold,
            ),
        }
    )

    predicted_px = norm_to_pixels(predictions[:, :2])
    target_px = norm_to_pixels(targets)
    confidence = (
        sigmoid(predictions[:, 2])
        if predictions.shape[1] == 3
        else np.ones(len(predictions))
    )
    output_rows: list[dict[str, str | float]] = []
    for index, sample_id in enumerate(sample_ids):
        distance: str | float = ""
        if has_track[index] == 1:
            distance = float(
                np.linalg.norm(predicted_px[index] - target_px[index])
            )
        output_rows.append(
            {
                "sample_id": sample_id,
                "video_id": video_ids[index],
                "image_path": image_paths[index],
                "has_track": int(has_track[index]),
                "pred_x_norm": float(predictions[index, 0]),
                "pred_y_norm": float(predictions[index, 1]),
                "pred_x_px": float(predicted_px[index, 0]),
                "pred_y_px": float(predicted_px[index, 1]),
                "confidence": float(confidence[index]),
                "target_x_px": (
                    float(target_px[index, 0]) if has_track[index] else ""
                ),
                "target_y_px": (
                    float(target_px[index, 1]) if has_track[index] else ""
                ),
                "distance_error_px": distance,
            }
        )

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", report)
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    render_largest_errors(output_rows, output_dir, args.render_largest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
