#!/usr/bin/env python3
"""Validate Task1 MP4/JSON pairs and create video_manifest.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2

from common import (
    relative_to_repo,
    resolve_repo_path,
    sha256_file,
)


FIELDS = [
    "video_id",
    "file_name",
    "metadata_file",
    "sha256",
    "task",
    "lap",
    "direction",
    "lighting",
    "obstacle_layout",
    "camera_mount",
    "width",
    "height",
    "fps",
    "frame_count",
    "duration_seconds",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="Task2/data/raw")
    parser.add_argument(
        "--output",
        default="Task2/data/video_manifest.csv",
    )
    parser.add_argument(
        "--existing-manifest",
        help="Optional manifest whose manually entered scene fields are retained.",
    )
    parser.add_argument(
        "--recording-manifest",
        help="Optional Task1 recording_manifest.csv used to import scene fields.",
    )
    return parser.parse_args()


def read_existing(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    manifest_path = resolve_repo_path(path)
    with manifest_path.open(newline="", encoding="utf-8") as file:
        return {
            row["video_id"]: row
            for row in csv.DictReader(file)
            if row.get("video_id")
        }


def read_recordings(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    manifest_path = resolve_repo_path(path)
    with manifest_path.open(newline="", encoding="utf-8") as file:
        return {
            Path(row["file_name"]).name: row
            for row in csv.DictReader(file)
            if row.get("file_name")
        }


def decode_probe(capture: cv2.VideoCapture, index: int) -> bool:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, frame = capture.read()
    return bool(ok and frame is not None and frame.size)


def inspect_video(video_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "status": "error",
            "notes": "OpenCV could not open video",
        }

    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = [0, frame_count // 2, max(0, frame_count - 1)]
    probes = [decode_probe(capture, index) for index in indices]
    capture.release()

    notes: list[str] = []
    if not all(probes):
        notes.append("first/middle/last decode probe failed")
    for field, observed in [
        ("width", width),
        ("height", height),
        ("frames_written", frame_count),
    ]:
        expected = metadata.get(field)
        if expected is not None and abs(int(expected) - observed) > 1:
            notes.append(f"metadata {field}={expected}, video={observed}")
    metadata_fps = metadata.get("container_fps")
    if (
        metadata_fps is not None
        and math.isfinite(fps)
        and abs(float(metadata_fps) - fps) > 0.1
    ):
        notes.append(f"metadata fps={metadata_fps}, video={fps:.3f}")

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / fps, 6) if fps > 0 else "",
        "status": "ready" if all(probes) else "error",
        "notes": "; ".join(notes),
    }


def main() -> int:
    args = parse_args()
    raw_dir = resolve_repo_path(args.raw_dir)
    output_path = resolve_repo_path(args.output)
    existing = read_existing(args.existing_manifest)
    recordings = read_recordings(args.recording_manifest)
    if not raw_dir.is_dir():
        print(f"Raw directory does not exist: {raw_dir}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for video_path in sorted(raw_dir.glob("*.mp4")):
        video_id = video_path.stem
        metadata_path = video_path.with_suffix(".json")
        row: dict[str, Any] = {field: "" for field in FIELDS}
        recording = recordings.get(video_path.name, {})
        row.update({field: recording[field] for field in FIELDS if field in recording})
        row.update(existing.get(video_id, {}))
        row.update(
            {
                "video_id": video_id,
                "file_name": relative_to_repo(video_path),
                "metadata_file": (
                    relative_to_repo(metadata_path)
                    if metadata_path.exists()
                    else ""
                ),
                "sha256": sha256_file(video_path),
            }
        )
        if not metadata_path.exists():
            row.update(status="error", notes="matching metadata JSON is missing")
        else:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                row.update(inspect_video(video_path, metadata))
            except (json.JSONDecodeError, OSError, ValueError) as error:
                row.update(status="error", notes=f"metadata error: {error}")
        if args.recording_manifest and not recording:
            existing_notes = str(row.get("notes", "")).strip()
            missing_note = "Task1 recording manifest row is missing"
            row["notes"] = "; ".join(
                note for note in [existing_notes, missing_note] if note
            )
            row["status"] = "error"
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    errors = sum(row["status"] != "ready" for row in rows)
    print(f"Wrote {len(rows)} video rows to {output_path}; errors={errors}")
    return 1 if not rows or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
