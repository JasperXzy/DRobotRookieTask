#!/usr/bin/env python3
"""Run a Task2 checkpoint on an MP4 and write a 640x480 overlay video."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common import load_yaml, resolve_repo_path, write_json
from dataset import preprocess_image
from metrics import norm_to_pixels, sigmoid
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--config",
        default=(
            "Task2/attachments/track_detection_resnet18/"
            "configs/train_resnet18_xyv.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        default="Task2/output/test_video.mp4",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(config["evaluation"]["confidence_threshold"])
    )
    device = torch.device(args.device)
    model, metadata = load_checkpoint(
        resolve_repo_path(args.checkpoint),
        device=device,
    )
    output_size = int(metadata.get("output_size", 3))
    video_path = resolve_repo_path(args.video)
    output_path = resolve_repo_path(args.output)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Could not open {video_path}", file=sys.stderr)
        return 1
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        print("Input video has invalid FPS.", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (640, 480),
    )
    if not writer.isOpened():
        capture.release()
        print(f"Could not create {output_path}", file=sys.stderr)
        return 1

    frame_count = 0
    detected_count = 0
    with torch.no_grad():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            standardized = cv2.resize(
                frame,
                (640, 480),
                interpolation=cv2.INTER_LINEAR,
            )
            roi = standardized[256:480, 0:640]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = preprocess_image(Image.fromarray(rgb), config)
            output = model(tensor.unsqueeze(0).to(device)).cpu().numpy()[0]
            pixel = norm_to_pixels(output[None, :2])[0]
            confidence = (
                float(sigmoid(np.asarray([output[2]]))[0])
                if output_size == 3
                else 1.0
            )
            detected = confidence >= threshold
            if detected:
                point = (
                    int(np.clip(round(pixel[0]), 0, 639)),
                    int(np.clip(round(pixel[1] + 256), 256, 479)),
                )
                cv2.circle(standardized, point, 8, (0, 0, 255), -1)
                detected_count += 1
            cv2.rectangle(standardized, (0, 256), (639, 479), (255, 180, 0), 1)
            cv2.putText(
                standardized,
                f"frame={frame_count} confidence={confidence:.3f}",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(standardized)
            frame_count += 1

    capture.release()
    writer.release()
    summary = {
        "input_video": str(video_path),
        "output_video": str(output_path),
        "checkpoint": str(resolve_repo_path(args.checkpoint)),
        "frame_count": frame_count,
        "detected_count": detected_count,
        "confidence_threshold": threshold,
    }
    write_json(output_path.with_suffix(".json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if frame_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
