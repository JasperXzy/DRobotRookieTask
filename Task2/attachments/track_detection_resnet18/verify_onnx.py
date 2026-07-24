#!/usr/bin/env python3
"""Compare Task2 PyTorch and ONNX Runtime outputs on a manifest split."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import onnx
import onnxruntime
import torch
from torch.utils.data import DataLoader

from common import load_yaml, resolve_repo_path, write_json
from dataset import TrackDataset
from metrics import norm_to_pixels
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
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
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--output",
        default="Task2/output/onnx_parity_report.json",
    )
    parser.add_argument("--raw-tolerance", type=float, default=1e-4)
    parser.add_argument("--pixel-tolerance", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    config = load_yaml(args.config)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    onnx_path = resolve_repo_path(args.onnx)
    model, metadata = load_checkpoint(checkpoint_path, device="cpu")
    output_size = int(metadata.get("output_size", 3))
    dataset = TrackDataset(
        resolve_repo_path(args.manifest),
        config,
        split=args.split,
        training=False,
        positive_only=output_size == 2,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    raw_differences: list[float] = []
    pixel_differences: list[float] = []
    invalid_count = 0
    sample_ids: list[str] = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.samples:
                break
            tensor = batch["image"].numpy().astype(np.float32, copy=False)
            torch_output = model(batch["image"]).numpy()
            onnx_output = session.run([output_name], {input_name: tensor})[0]
            if not np.isfinite(torch_output).all() or not np.isfinite(
                onnx_output
            ).all():
                invalid_count += 1
            raw_differences.append(float(np.max(np.abs(torch_output - onnx_output))))
            torch_px = norm_to_pixels(torch_output[:, :2])
            onnx_px = norm_to_pixels(onnx_output[:, :2])
            pixel_differences.append(
                float(np.max(np.abs(torch_px - onnx_px)))
            )
            sample_ids.extend(batch["sample_id"])

    tested = len(raw_differences)
    max_raw = max(raw_differences, default=float("inf"))
    max_pixel = max(pixel_differences, default=float("inf"))
    passed = (
        tested >= args.samples
        and invalid_count == 0
        and max_raw <= args.raw_tolerance
        and max_pixel <= args.pixel_tolerance
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "split": args.split,
        "requested_samples": args.samples,
        "tested_samples": tested,
        "sample_ids": sample_ids,
        "onnx_checker": "passed",
        "invalid_output_count": invalid_count,
        "max_raw_absolute_difference": max_raw,
        "max_coordinate_difference_px": max_pixel,
        "raw_tolerance": args.raw_tolerance,
        "pixel_tolerance": args.pixel_tolerance,
        "passed": passed,
    }
    write_json(resolve_repo_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
