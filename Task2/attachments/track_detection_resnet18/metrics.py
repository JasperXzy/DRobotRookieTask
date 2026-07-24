#!/usr/bin/env python3
"""Task2 coordinate and track-confidence metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def norm_to_pixels(xy: np.ndarray) -> np.ndarray:
    result = np.asarray(xy, dtype=np.float64).copy()
    result[..., 0] = (result[..., 0] + 1.0) * 320.0
    result[..., 1] = (result[..., 1] + 1.0) * 112.0
    return result


def compute_metrics(
    predictions: np.ndarray,
    targets_xy: np.ndarray,
    has_track: np.ndarray,
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets_xy = np.asarray(targets_xy, dtype=np.float64)
    has_track = np.asarray(has_track, dtype=np.int64)
    positive = has_track == 1

    report: dict[str, Any] = {
        "sample_count": int(len(has_track)),
        "positive_count": int(positive.sum()),
        "negative_count": int((has_track == 0).sum()),
    }
    if positive.any():
        predicted_px = norm_to_pixels(predictions[positive, :2])
        target_px = norm_to_pixels(targets_xy[positive])
        delta = np.abs(predicted_px - target_px)
        distance = np.linalg.norm(predicted_px - target_px, axis=1)
        report.update(
            {
                "x_mae_px": float(delta[:, 0].mean()),
                "y_mae_px": float(delta[:, 1].mean()),
                "distance_mae_px": float(distance.mean()),
                "distance_p90_px": float(np.percentile(distance, 90)),
                "distance_p95_px": float(np.percentile(distance, 95)),
                "hit_rate_10px": float((distance <= 10).mean()),
                "hit_rate_20px": float((distance <= 20).mean()),
                "hit_rate_30px": float((distance <= 30).mean()),
            }
        )

    if predictions.shape[1] >= 3:
        confidence = sigmoid(predictions[:, 2])
        predicted_track = confidence >= confidence_threshold
        truth = has_track == 1
        true_positive = int(np.logical_and(predicted_track, truth).sum())
        false_positive = int(np.logical_and(predicted_track, ~truth).sum())
        false_negative = int(np.logical_and(~predicted_track, truth).sum())
        true_negative = int(np.logical_and(~predicted_track, ~truth).sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        report.update(
            {
                "confidence_threshold": confidence_threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": {
                    "tp": true_positive,
                    "fp": false_positive,
                    "fn": false_negative,
                    "tn": true_negative,
                },
            }
        )
    return report
