#!/usr/bin/env python3
"""Assign whole video_id groups to train, val, and test splits."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import load_yaml, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="Task2/data/dataset_manifest.csv",
    )
    parser.add_argument(
        "--config",
        default=(
            "Task2/attachments/track_detection_resnet18/"
            "configs/dataset.yaml"
        ),
    )
    parser.add_argument(
        "--split-manifest",
        default="Task2/data/split_manifest.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing split assignment.",
    )
    return parser.parse_args()


def allocate_groups(
    video_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    if len(video_ids) < 3:
        raise ValueError(
            "at least three video_id groups are required to keep "
            "train, val, and test non-empty"
        )
    shuffled = sorted(video_ids)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    test_count = max(1, round(count * (1.0 - train_ratio - val_ratio)))
    val_count = max(1, round(count * val_ratio))
    if test_count + val_count >= count:
        test_count = 1
        val_count = 1
    test_ids = set(shuffled[:test_count])
    val_ids = set(shuffled[test_count : test_count + val_count])
    return {
        video_id: (
            "test"
            if video_id in test_ids
            else "val"
            if video_id in val_ids
            else "train"
        )
        for video_id in shuffled
    }


def main() -> int:
    args = parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    split_path = resolve_repo_path(args.split_manifest)
    config = load_yaml(args.config)["split"]
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        print("Dataset manifest is empty.", file=sys.stderr)
        return 1
    if "split" not in fieldnames:
        fieldnames.append("split")
    if not args.force and any(row.get("split") for row in rows):
        print(
            "Split assignments already exist; pass --force to replace them.",
            file=sys.stderr,
        )
        return 2

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("annotation_status") in {"positive", "negative"}:
            grouped[row["video_id"]].append(row)
    assignments = allocate_groups(
        list(grouped),
        float(config["train_ratio"]),
        float(config["val_ratio"]),
        int(config["seed"]),
    )
    for row in rows:
        row["split"] = assignments.get(row["video_id"], "")

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_fields = [
        "video_id",
        "recording_session",
        "split",
        "direction",
        "lighting",
        "task",
        "frame_count",
        "positive_count",
        "negative_count",
        "reason",
        "locked",
    ]
    split_rows: list[dict[str, Any]] = []
    for video_id, video_rows in sorted(grouped.items()):
        counts = Counter(row["annotation_status"] for row in video_rows)
        first = video_rows[0]
        split_rows.append(
            {
                "video_id": video_id,
                "recording_session": video_id,
                "split": assignments[video_id],
                "direction": first.get("direction", ""),
                "lighting": first.get("lighting", ""),
                "task": first.get("task", ""),
                "frame_count": len(video_rows),
                "positive_count": counts["positive"],
                "negative_count": counts["negative"],
                "reason": "deterministic video-level split",
                "locked": "true" if assignments[video_id] == "test" else "false",
            }
        )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=split_fields)
        writer.writeheader()
        writer.writerows(split_rows)

    counts = Counter(assignments.values())
    print(f"Wrote {split_path}: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
