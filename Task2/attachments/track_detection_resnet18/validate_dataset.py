#!/usr/bin/env python3
"""Validate Task2 TXT labels and build dataset_manifest.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from common import load_yaml, resolve_repo_path, write_json


FIELDS = [
    "sample_id",
    "video_id",
    "frame_index",
    "timestamp_ms",
    "image_path",
    "task",
    "lap",
    "direction",
    "lighting",
    "quality_status",
    "x_roi",
    "y_roi",
    "x_norm",
    "y_norm",
    "has_track",
    "annotation_status",
    "reviewer",
    "split",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame-manifest",
        default="Task2/data/frame_manifest.csv",
    )
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
        "--output",
        default="Task2/data/dataset_manifest.csv",
    )
    parser.add_argument(
        "--label-dir",
        help="Override config paths.annotations_dir.",
    )
    parser.add_argument(
        "--report",
        default="Task2/output/dataset_validation.json",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail when any frame is still unlabeled.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_label(
    path: Path,
    width: int,
    height: int,
) -> tuple[str, str, str, str, str, str]:
    if not path.exists():
        return "", "", "", "", "", "unlabeled"
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 3:
        raise ValueError("expected exactly three fields: x_roi y_roi state")
    state = int(fields[2])
    if state == 1:
        x = float(fields[0])
        y = float(fields[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("positive coordinates must be finite")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"point ({x}, {y}) outside {width}x{height}")
        x_norm = 2.0 * x / width - 1.0
        y_norm = 2.0 * y / height - 1.0
        return (
            f"{x:.6f}",
            f"{y:.6f}",
            f"{x_norm:.9f}",
            f"{y_norm:.9f}",
            "1",
            "positive",
        )
    if state == 0:
        if not all(value.lower() == "nan" for value in fields[:2]):
            raise ValueError("negative label must be 'nan nan 0'")
        return "", "", "", "", "0", "negative"
    if state == -1:
        if not all(value.lower() == "nan" for value in fields[:2]):
            raise ValueError("rejected label must be 'nan nan -1'")
        return "", "", "", "", "", "rejected"
    raise ValueError(f"unsupported state {state}")


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    geometry = config["geometry"]
    width = int(geometry["roi_width"])
    height = int(geometry["roi_height"])
    label_dir = resolve_repo_path(
        args.label_dir or config["paths"]["annotations_dir"]
    )
    frame_manifest = resolve_repo_path(args.frame_manifest)
    video_manifest = resolve_repo_path(args.video_manifest)
    output_path = resolve_repo_path(args.output)
    report_path = resolve_repo_path(args.report)

    videos = {
        row["video_id"]: row
        for row in read_csv(video_manifest)
        if row.get("video_id")
    }
    previous: dict[str, dict[str, str]] = {}
    if output_path.exists():
        previous = {
            row["sample_id"]: row
            for row in read_csv(output_path)
            if row.get("sample_id")
        }

    output_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for frame in read_csv(frame_manifest):
        sample_id = frame["sample_id"]
        video = videos.get(frame["video_id"], {})
        label_path = label_dir / f"{sample_id}.txt"
        try:
            x, y, x_norm, y_norm, has_track, status = parse_label(
                label_path,
                width,
                height,
            )
        except (OSError, ValueError) as error:
            errors.append({"sample_id": sample_id, "error": str(error)})
            x = y = x_norm = y_norm = has_track = ""
            status = "invalid"

        old = previous.get(sample_id, {})
        counts[status] += 1
        output_rows.append(
            {
                "sample_id": sample_id,
                "video_id": frame["video_id"],
                "frame_index": frame["frame_index"],
                "timestamp_ms": frame["timestamp_ms"],
                "image_path": frame["roi_image_path"],
                "task": video.get("task", ""),
                "lap": video.get("lap", ""),
                "direction": video.get("direction", ""),
                "lighting": video.get("lighting", ""),
                "quality_status": frame.get("quality_status", ""),
                "x_roi": x,
                "y_roi": y,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "has_track": has_track,
                "annotation_status": status,
                "reviewer": old.get("reviewer", ""),
                "split": old.get("split", ""),
                "notes": frame.get("quality_flags", ""),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "frame_manifest": str(frame_manifest),
        "dataset_manifest": str(output_path),
        "sample_count": len(output_rows),
        "status_counts": dict(sorted(counts.items())),
        "error_count": len(errors),
        "errors": errors[:100],
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    incomplete = counts["unlabeled"] if args.require_complete else 0
    if errors or incomplete:
        print("Dataset validation failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
