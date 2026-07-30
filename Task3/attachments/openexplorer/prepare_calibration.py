#!/usr/bin/env python3
"""Build representative RGB CHW float32 PTQ calibration binaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


MODEL_SIZE = (224, 224)
EXPECTED_BYTES = 3 * MODEL_SIZE[0] * MODEL_SIZE[1] * np.dtype(np.float32).itemsize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing calibration files with matching generated names.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_repo_root(manifest: Path) -> Path:
    for parent in (manifest.parent, *manifest.parents):
        if (parent / "Task2").is_dir() and (parent / "Task3").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repository root from {manifest}")


def resolve_image_path(raw_path: str, repo_root: Path, manifest_dir: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate
    return manifest_dir / candidate


def normalized_status(row: dict[str, str]) -> str:
    status = (row.get("annotation_status") or "").strip().lower()
    if status:
        return status
    has_track = (row.get("has_track") or "").strip().lower()
    if has_track in {"1", "true", "yes"}:
        return "positive"
    if has_track in {"0", "false", "no"}:
        return "negative"
    return "unlabeled"


def stratum(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        (row.get("video_id") or "unknown_video").strip(),
        (row.get("direction") or "unknown_direction").strip(),
        (row.get("lighting") or "unknown_lighting").strip(),
        normalized_status(row),
    )


def safe_sample_id(row: dict[str, str], index: int) -> str:
    return (
        (row.get("sample_id") or row.get("frame_id") or f"sample_{index:04d}")
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
    )


def select_balanced(
    rows: list[dict[str, str]],
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[stratum(row)].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    ordered_keys = sorted(buckets)
    rng.shuffle(ordered_keys)
    queues = {key: deque(buckets[key]) for key in ordered_keys}
    selected: list[dict[str, str]] = []
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            if queues[key]:
                selected.append(queues[key].popleft())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def read_rows(manifest: Path, allowed_splits: set[str]) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError("Manifest must contain an image_path column.")
        rows = [
            dict(row)
            for row in reader
            if (row.get("split") or "").strip().lower() in allowed_splits
        ]
    if not rows:
        raise ValueError(f"No rows matched splits: {sorted(allowed_splits)}")
    return rows


def calibration_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(MODEL_SIZE, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Refusing to write an empty calibration manifest.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive.")

    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selection_manifest = args.selection_manifest.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    existing_rgb = list(output_dir.glob("*.rgb")) if output_dir.exists() else []
    if existing_rgb and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing_rgb)} .rgb files. "
            "Use --overwrite only when deliberately regenerating them."
        )

    repo_root = find_repo_root(manifest)
    allowed_splits = {item.strip().lower() for item in args.splits}
    candidates = read_rows(manifest, allowed_splits)
    if args.count > len(candidates):
        raise ValueError(
            f"Requested {args.count} samples, but only {len(candidates)} candidates exist."
        )
    candidate_status_counts = Counter(normalized_status(row) for row in candidates)
    for required_status in ("positive", "negative"):
        if candidate_status_counts.get(required_status, 0) == 0:
            raise ValueError(
                f"Final acceptance requires {required_status} train/val samples."
            )
    selected = select_balanced(candidates, args.count, args.seed)
    selected_status_counts = Counter(normalized_status(row) for row in selected)
    for required_status in ("positive", "negative"):
        if selected_status_counts.get(required_status, 0) == 0:
            raise ValueError(
                f"The selected calibration set contains no {required_status} samples."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {
        f"{index:04d}_{safe_sample_id(row, index)}.rgb"
        for index, row in enumerate(selected)
    }
    stale_names = {path.name for path in existing_rgb} - expected_names
    if stale_names:
        preview = ", ".join(sorted(stale_names)[:5])
        raise FileExistsError(
            f"{output_dir} contains stale calibration files ({preview}). "
            "Use a new empty output directory; stale files must not be mixed "
            "into calibration."
        )

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        source_path = resolve_image_path(
            row["image_path"],
            repo_root,
            manifest.parent,
        ).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Calibration source image not found: {source_path}")

        sample_id = safe_sample_id(row, index)
        output_path = output_dir / f"{index:04d}_{sample_id}.rgb"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Calibration file already exists: {output_path}")

        array = calibration_array(source_path)
        if array.shape != (3, 224, 224) or array.dtype != np.float32:
            raise RuntimeError(
                f"Unexpected calibration tensor for {source_path}: "
                f"{array.shape} {array.dtype}"
            )
        array.tofile(output_path)
        if output_path.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError(f"Unexpected calibration file size: {output_path}")

        output_rows.append(
            {
                "index": index,
                "sample_id": sample_id,
                "split": (row.get("split") or "").strip(),
                "video_id": (row.get("video_id") or "").strip(),
                "direction": (row.get("direction") or "").strip(),
                "lighting": (row.get("lighting") or "").strip(),
                "annotation_status": normalized_status(row),
                "source_image": str(source_path),
                "calibration_file": str(output_path),
                "shape": "3x224x224",
                "dtype": "float32",
                "value_range": "0..255",
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256(output_path),
            }
        )

    write_csv(selection_manifest, output_rows)
    status_counts = Counter(row["annotation_status"] for row in output_rows)
    split_counts = Counter(row["split"] for row in output_rows)
    stratum_counts = Counter(
        "|".join(
            [
                row["video_id"],
                row["direction"],
                row["lighting"],
                row["annotation_status"],
            ]
        )
        for row in output_rows
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "seed": args.seed,
        "allowed_splits": sorted(allowed_splits),
        "candidate_count": len(candidates),
        "selected_count": len(output_rows),
        "tensor_contract": {
            "color": "RGB",
            "layout": "CHW",
            "shape": [3, 224, 224],
            "dtype": "float32",
            "value_range": [0.0, 255.0],
            "normalization": "none; OpenExplorer applies mean and scale",
            "bytes_per_sample": EXPECTED_BYTES,
        },
        "counts": {
            "split": dict(sorted(split_counts.items())),
            "annotation_status": dict(sorted(status_counts.items())),
            "stratum": dict(sorted(stratum_counts.items())),
        },
        "warnings": [],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(output_rows)} calibration samples to {output_dir}")
    print(f"Selection manifest: {selection_manifest}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
