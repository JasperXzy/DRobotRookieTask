#!/usr/bin/env python3
"""Manifest-backed Task2 dataset and fixed model preprocessing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common import resolve_data_path


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
ROI_WIDTH = 640
ROI_HEIGHT = 224


def make_transform(config: dict[str, Any], training: bool) -> transforms.Compose:
    input_config = config["input"]
    steps: list[Any] = [
        transforms.Resize(
            (int(input_config["height"]), int(input_config["width"])),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
    ]
    if training:
        augmentation = config.get("augmentation", {})
        jitter_values = [
            float(augmentation.get("brightness", 0.0)),
            float(augmentation.get("contrast", 0.0)),
            float(augmentation.get("saturation", 0.0)),
            float(augmentation.get("hue", 0.0)),
        ]
        if any(jitter_values):
            steps.append(transforms.ColorJitter(*jitter_values))
        blur_probability = float(
            augmentation.get("gaussian_blur_probability", 0.0)
        )
        if blur_probability > 0:
            steps.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)],
                    p=blur_probability,
                )
            )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=input_config.get("mean", DEFAULT_MEAN),
                std=input_config.get("std", DEFAULT_STD),
            ),
        ]
    )
    return transforms.Compose(steps)


def preprocess_image(
    image: Image.Image,
    config: dict[str, Any],
) -> torch.Tensor:
    return make_transform(config, training=False)(image.convert("RGB"))


class TrackDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        config: dict[str, Any],
        split: str,
        training: bool = False,
        positive_only: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        frame = pd.read_csv(self.manifest_path)
        required = {"sample_id", "image_path", "split", "has_track"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{self.manifest_path} is missing columns: {sorted(missing)}"
            )

        frame = frame[frame["split"].astype(str) == split].copy()
        frame["has_track"] = pd.to_numeric(frame["has_track"], errors="coerce")
        frame = frame[frame["has_track"].isin([0, 1])]
        if "annotation_status" in frame.columns:
            frame = frame[
                ~frame["annotation_status"].astype(str).isin(
                    ["unlabeled", "rejected"]
                )
            ]
        if positive_only:
            frame = frame[frame["has_track"] == 1]
        if frame.empty:
            raise ValueError(f"no usable samples for split={split!r}")

        self.rows = frame.reset_index(drop=True)
        self.transform = make_transform(config, training=training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        image_path = resolve_data_path(str(row["image_path"]))
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))

        has_track = float(row["has_track"])
        if has_track == 1:
            x_norm = _coordinate(row, "x_norm", "x_roi", ROI_WIDTH)
            y_norm = _coordinate(row, "y_norm", "y_roi", ROI_HEIGHT)
        else:
            x_norm = 0.0
            y_norm = 0.0

        return {
            "image": tensor,
            "xy": torch.tensor([x_norm, y_norm], dtype=torch.float32),
            "has_track": torch.tensor(has_track, dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "image_path": str(image_path),
            "video_id": str(row.get("video_id", "")),
            "direction": str(row.get("direction", "")),
            "lighting": str(row.get("lighting", "")),
            "task": str(row.get("task", "")),
        }


def _coordinate(
    row: pd.Series,
    normalized_name: str,
    pixel_name: str,
    size: int,
) -> float:
    value = row.get(normalized_name)
    if value is not None and not pd.isna(value):
        normalized = float(value)
    else:
        pixel = float(row[pixel_name])
        normalized = 2.0 * pixel / float(size) - 1.0
    if not math.isfinite(normalized):
        raise ValueError(f"{row['sample_id']}: {normalized_name} is not finite")
    return normalized
