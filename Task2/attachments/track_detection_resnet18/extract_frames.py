#!/usr/bin/env python3
"""Sample Task1 videos, resize to 640x480, and crop the 640x224 ROI."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import (
    load_yaml,
    relative_to_task2_data,
    resolve_repo_path,
)


FIELDS = [
    "sample_id",
    "video_id",
    "frame_index",
    "timestamp_ms",
    "full_image_path",
    "roi_image_path",
    "quality_status",
    "quality_flags",
    "blur_score",
    "brightness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-manifest",
        default="Task2/data/video_manifest.csv",
    )
    parser.add_argument(
        "--config",
        default=(
            "Task2/attachments/track_detection_resnet18/"
            "configs/dataset.yaml"
        ),
    )
    parser.add_argument(
        "--output-manifest",
        default="Task2/data/frame_manifest.csv",
    )
    parser.add_argument(
        "--full-dir",
        help="Override config paths.full_frames_dir.",
    )
    parser.add_argument(
        "--roi-dir",
        help="Override config paths.roi_frames_dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already extracted image files.",
    )
    return parser.parse_args()


def perceptual_hash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return small >= small.mean()


def quality(
    frame: np.ndarray,
    previous_hash: np.ndarray | None,
) -> tuple[str, str, float, float, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    current_hash = perceptual_hash(gray)
    flags: list[str] = []
    if blur_score < 40:
        flags.append("blur")
    if brightness < 35:
        flags.append("underexposed")
    if brightness > 225:
        flags.append("overexposed")
    difference = (
        int(np.count_nonzero(current_hash != previous_hash))
        if previous_hash is not None
        else None
    )
    if difference is not None and difference <= 3:
        flags.append("near_duplicate")
    return (
        "suspect" if flags else "accepted",
        "|".join(flags),
        blur_score,
        brightness,
        current_hash,
    )


def open_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if row.get("status") == "ready"]


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    paths = config["paths"]
    sampling = config["sampling"]
    geometry = config["geometry"]
    video_manifest = resolve_repo_path(args.video_manifest)
    output_manifest = resolve_repo_path(args.output_manifest)
    full_dir = resolve_repo_path(args.full_dir or paths["full_frames_dir"])
    roi_dir = resolve_repo_path(args.roi_dir or paths["roi_frames_dir"])
    full_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    target_fps = float(sampling["sample_fps"])
    source_size = (
        int(geometry["source_width"]),
        int(geometry["source_height"]),
    )
    roi_x = int(geometry["roi_x"])
    roi_y = int(geometry["roi_y"])
    roi_width = int(geometry["roi_width"])
    roi_height = int(geometry["roi_height"])
    extension = str(sampling.get("image_extension", ".jpg"))
    jpeg_quality = int(sampling.get("jpeg_quality", 95))

    if target_fps <= 0:
        raise ValueError("sampling.sample_fps must be positive")

    rows: list[dict[str, Any]] = []
    for video_row in open_manifest(video_manifest):
        video_id = video_row["video_id"]
        video_path = resolve_repo_path(video_row["file_name"])
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"Cannot open {video_path}", file=sys.stderr)
            continue
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0:
            print(f"Invalid FPS for {video_path}", file=sys.stderr)
            capture.release()
            continue

        next_sample_time = 0.0
        frame_index = 0
        previous_hash: np.ndarray | None = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            if timestamp + 1e-9 < next_sample_time:
                frame_index += 1
                continue
            next_sample_time += 1.0 / target_fps

            resized = cv2.resize(frame, source_size, interpolation=cv2.INTER_LINEAR)
            roi = resized[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
            if roi.shape[:2] != (roi_height, roi_width):
                capture.release()
                raise ValueError(f"Invalid ROI configuration produced {roi.shape}")

            sample_id = f"{video_id}__f{frame_index:06d}"
            full_path = full_dir / f"{sample_id}{extension}"
            roi_path = roi_dir / f"{sample_id}{extension}"
            write_options = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            if args.overwrite or not full_path.exists():
                if not cv2.imwrite(str(full_path), resized, write_options):
                    raise OSError(f"Could not write {full_path}")
            if args.overwrite or not roi_path.exists():
                if not cv2.imwrite(str(roi_path), roi, write_options):
                    raise OSError(f"Could not write {roi_path}")

            quality_status, flags, blur, brightness, previous_hash = quality(
                roi,
                previous_hash,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "timestamp_ms": round(timestamp * 1000.0, 3),
                    "full_image_path": relative_to_task2_data(full_path),
                    "roi_image_path": relative_to_task2_data(roi_path),
                    "quality_status": quality_status,
                    "quality_flags": flags,
                    "blur_score": round(blur, 3),
                    "brightness": round(brightness, 3),
                }
            )
            frame_index += 1
        capture.release()

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sampled frames to {output_manifest}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
