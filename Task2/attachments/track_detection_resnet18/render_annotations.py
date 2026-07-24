#!/usr/bin/env python3
"""Render a contact sheet for Task2 annotations."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from common import resolve_data_path, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="Task2/data/dataset_manifest.csv",
    )
    parser.add_argument(
        "--output",
        default="Task2/output/annotation_contact_sheet.jpg",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row.get("annotation_status") in {"positive", "negative", "rejected"}
        ]
    if not rows:
        print("No labeled samples found.", file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(rows)
    rows = rows[: max(1, args.limit)]
    tile_width = 320
    image_height = 112
    header_height = 28
    tile_height = header_height + image_height
    columns = max(1, args.columns)
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = np.zeros(
        (sheet_rows * tile_height, columns * tile_width, 3),
        dtype=np.uint8,
    )

    for index, row in enumerate(rows):
        image_path = resolve_data_path(row["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not read {image_path}", file=sys.stderr)
            return 1
        image = cv2.resize(
            image,
            (tile_width, image_height),
            interpolation=cv2.INTER_AREA,
        )
        status = row["annotation_status"]
        color = {
            "positive": (0, 255, 0),
            "negative": (0, 180, 255),
            "rejected": (0, 0, 255),
        }[status]
        if status == "positive":
            x = round(float(row["x_roi"]) * tile_width / 640.0)
            y = round(float(row["y_roi"]) * image_height / 224.0)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        tile[header_height:] = image
        text = f"{row['sample_id']} {status}"
        cv2.putText(
            tile,
            text[:50],
            (4, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        top = (index // columns) * tile_height
        left = (index % columns) * tile_width
        sheet[top : top + tile_height, left : left + tile_width] = tile

    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        print(f"Could not write {output_path}", file=sys.stderr)
        return 1
    print(f"Wrote {len(rows)} annotations to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
