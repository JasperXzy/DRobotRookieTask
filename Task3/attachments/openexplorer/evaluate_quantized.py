#!/usr/bin/env python3
"""Compare Task2 float ONNX and OpenExplorer quantized ONNX on one split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float-model", type=Path, required=True)
    parser.add_argument("--quantized-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--mean-drift-limit-px", type=float, default=2.0)
    parser.add_argument("--p95-drift-limit-px", type=float, default=5.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_root_from(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "Task2").is_dir() and (parent / "Task3").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repository root from {path}")


def resolve_image(raw_path: str, manifest: Path, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate
    return manifest.parent / path


def parse_optional_float(row: dict[str, str], name: str) -> float | None:
    raw = (row.get(name) or "").strip()
    if not raw:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def normalize_coordinate(
    row: dict[str, str],
    normalized_name: str,
    pixel_name: str,
    size: int,
) -> float:
    normalized = parse_optional_float(row, normalized_name)
    if normalized is not None:
        return normalized
    pixel = parse_optional_float(row, pixel_name)
    if pixel is None:
        raise ValueError(
            f"{row.get('sample_id')}: missing {normalized_name} and {pixel_name}"
        )
    return 2.0 * pixel / float(size) - 1.0


def usable_rows(
    manifest: Path,
    split: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    repo_root = repo_root_from(manifest)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for source in reader:
            if (source.get("split") or "").strip() != split:
                continue
            status = (source.get("annotation_status") or "").strip().lower()
            if status in {"unlabeled", "rejected"}:
                continue
            has_track_raw = (source.get("has_track") or "").strip()
            if has_track_raw not in {"0", "1"}:
                continue
            has_track = int(has_track_raw)
            target = [0.0, 0.0]
            if has_track:
                target = [
                    normalize_coordinate(source, "x_norm", "x_roi", 640),
                    normalize_coordinate(source, "y_norm", "y_roi", 224),
                ]
            image_path = resolve_image(
                source["image_path"],
                manifest,
                repo_root,
            ).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            rows.append(
                {
                    **source,
                    "has_track_int": has_track,
                    "target_xy": target,
                    "resolved_image_path": image_path,
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No usable samples found for split={split!r}")
    return rows


def preprocess_float(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(
            (224, 224),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32)
    nchw = np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...] / 255.0)
    return (nchw - MEAN) / STD


def build_quantized_preprocessor(input_layout: str):
    from horizon_tc_ui.data.dataloader import SingleImageDataLoader
    from horizon_tc_ui.data.transformer import (
        BGR2NV12Transformer,
        NV12ToYUV444Transformer,
        ResizeTransformer,
    )

    if input_layout not in {"NCHW", "NHWC"}:
        raise ValueError(f"Unsupported quantized model input layout: {input_layout}")
    transformers = [
        ResizeTransformer(target_size=(224, 224)),
        BGR2NV12Transformer(data_format="HWC"),
        NV12ToYUV444Transformer(
            target_size=(224, 224),
            yuv444_output_layout=input_layout[1:],
        ),
    ]

    def preprocess(image_path: Path):
        return SingleImageDataLoader(
            transformers,
            str(image_path),
            imread_mode="opencv",
        )

    return preprocess


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def norm_to_pixels(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result[..., 0] = (result[..., 0] + 1.0) * 320.0
    result[..., 1] = (result[..., 1] + 1.0) * 112.0
    return result


def prediction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    has_track: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    positive = has_track == 1
    report: dict[str, Any] = {
        "sample_count": int(len(has_track)),
        "positive_count": int(positive.sum()),
        "negative_count": int((has_track == 0).sum()),
    }
    if positive.any():
        predicted_px = norm_to_pixels(predictions[positive, :2])
        target_px = norm_to_pixels(targets[positive])
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
        predicted_track = confidence >= threshold
        truth = has_track == 1
        tp = int(np.logical_and(predicted_track, truth).sum())
        fp = int(np.logical_and(predicted_track, ~truth).sum())
        fn = int(np.logical_and(~predicted_track, truth).sum())
        tn = int(np.logical_and(~predicted_track, ~truth).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        report.update(
            {
                "confidence_threshold": threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(
                    2.0 * precision * recall / max(1e-12, precision + recall)
                ),
                "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            }
        )
    return report


def grouped_comparison(
    rows: list[dict[str, Any]],
    field: str,
    float_predictions: np.ndarray,
    quantized_predictions: np.ndarray,
    targets: np.ndarray,
    has_track: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    values = np.asarray(
        [(row.get(field) or "").strip() for row in rows],
        dtype=object,
    )
    groups: dict[str, Any] = {}
    for value in sorted(set(values.tolist())):
        if not value:
            continue
        mask = values == value
        float_group = float_predictions[mask]
        quantized_group = quantized_predictions[mask]
        group_drift = np.linalg.norm(
            norm_to_pixels(quantized_group[:, :2])
            - norm_to_pixels(float_group[:, :2]),
            axis=1,
        )
        groups[value] = {
            "float": prediction_metrics(
                float_group,
                targets[mask],
                has_track[mask],
                threshold,
            ),
            "quantized": prediction_metrics(
                quantized_group,
                targets[mask],
                has_track[mask],
                threshold,
            ),
            "quantization_drift": {
                "mean_px": float(group_drift.mean()),
                "p95_px": float(np.percentile(group_drift, 95)),
                "max_px": float(group_drift.max()),
            },
        }
    return groups


def run_inference(
    float_model: Path,
    quantized_model: Path,
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort
    from horizon_tc_ui import HB_ONNXRuntime

    float_session = ort.InferenceSession(
        str(float_model),
        providers=["CPUExecutionProvider"],
    )
    if len(float_session.get_inputs()) != 1:
        raise ValueError("Float ONNX must have exactly one input.")
    float_input_name = float_session.get_inputs()[0].name
    float_output_name = float_session.get_outputs()[0].name

    quantized_session = HB_ONNXRuntime(model_file=str(quantized_model))
    input_layout = quantized_session.layout[0] or "NHWC"
    quantized_preprocess = build_quantized_preprocessor(input_layout)

    float_outputs: list[np.ndarray] = []
    quantized_outputs: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        image_path = row["resolved_image_path"]
        float_value = float_session.run(
            [float_output_name],
            {float_input_name: preprocess_float(image_path)},
        )[0]
        quantized_input = quantized_preprocess(image_path)
        quantized_value = quantized_session.run(
            quantized_session.output_names,
            {quantized_session.input_names[0]: quantized_input},
        )[0]
        float_vector = np.asarray(float_value, dtype=np.float32).reshape(-1)
        quantized_vector = np.asarray(quantized_value, dtype=np.float32).reshape(-1)
        if float_vector.size < 3 or quantized_vector.size < 3:
            raise ValueError(
                "Both models must expose at least three outputs in "
                "[x_norm, y_norm, track_logit] order."
            )
        float_outputs.append(float_vector[:3])
        quantized_outputs.append(quantized_vector[:3])
        if index % 25 == 0 or index == len(rows):
            print(f"Evaluated {index}/{len(rows)} samples")
    return np.stack(float_outputs), np.stack(quantized_outputs)


def write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    float_predictions: np.ndarray,
    quantized_predictions: np.ndarray,
) -> None:
    float_px = norm_to_pixels(float_predictions[:, :2])
    quantized_px = norm_to_pixels(quantized_predictions[:, :2])
    target_px = norm_to_pixels(np.asarray([row["target_xy"] for row in rows]))
    float_confidence = sigmoid(float_predictions[:, 2])
    quantized_confidence = sigmoid(quantized_predictions[:, 2])

    materialized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        drift = float(np.linalg.norm(quantized_px[index] - float_px[index]))
        has_track = int(row["has_track_int"])
        materialized.append(
            {
                "sample_id": row.get("sample_id", ""),
                "video_id": row.get("video_id", ""),
                "image_path": str(row["resolved_image_path"]),
                "has_track": has_track,
                "float_x_norm": float(float_predictions[index, 0]),
                "float_y_norm": float(float_predictions[index, 1]),
                "float_track_logit": float(float_predictions[index, 2]),
                "float_x_px": float(float_px[index, 0]),
                "float_y_px": float(float_px[index, 1]),
                "quantized_x_norm": float(quantized_predictions[index, 0]),
                "quantized_y_norm": float(quantized_predictions[index, 1]),
                "quantized_track_logit": float(quantized_predictions[index, 2]),
                "quantized_x_px": float(quantized_px[index, 0]),
                "quantized_y_px": float(quantized_px[index, 1]),
                "quantization_drift_px": drift,
                "float_confidence": float(float_confidence[index]),
                "quantized_confidence": float(quantized_confidence[index]),
                "target_x_px": float(target_px[index, 0]) if has_track else "",
                "target_y_px": float(target_px[index, 1]) if has_track else "",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def main() -> int:
    args = parse_args()
    float_model = args.float_model.expanduser().resolve()
    quantized_model = args.quantized_model.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (float_model, quantized_model, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = usable_rows(manifest, args.split, args.max_samples)
    targets = np.asarray([row["target_xy"] for row in rows], dtype=np.float32)
    has_track = np.asarray([row["has_track_int"] for row in rows], dtype=np.int64)
    float_predictions, quantized_predictions = run_inference(
        float_model,
        quantized_model,
        rows,
    )
    finite_predictions = bool(
        np.isfinite(float_predictions).all()
        and np.isfinite(quantized_predictions).all()
    )

    float_metrics = prediction_metrics(
        float_predictions,
        targets,
        has_track,
        args.confidence_threshold,
    )
    quantized_metrics = prediction_metrics(
        quantized_predictions,
        targets,
        has_track,
        args.confidence_threshold,
    )
    float_px = norm_to_pixels(float_predictions[:, :2])
    quantized_px = norm_to_pixels(quantized_predictions[:, :2])
    drift = np.linalg.norm(quantized_px - float_px, axis=1)

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, limit: Any) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "actual": actual, "limit": limit}
        )

    check("finite_predictions", finite_predictions, finite_predictions, True)
    check(
        "minimum_sample_count",
        len(rows) >= args.min_samples,
        len(rows),
        f">={args.min_samples}",
    )
    check(
        "mean_quantization_drift_px",
        float(drift.mean()) <= args.mean_drift_limit_px,
        float(drift.mean()),
        f"<={args.mean_drift_limit_px}",
    )
    check(
        "p95_quantization_drift_px",
        float(np.percentile(drift, 95)) <= args.p95_drift_limit_px,
        float(np.percentile(drift, 95)),
        f"<={args.p95_drift_limit_px}",
    )
    if has_track.any():
        float_mae = float(float_metrics["distance_mae_px"])
        quantized_mae = float(quantized_metrics["distance_mae_px"])
        allowed_mae = float_mae + max(2.0, float_mae * 0.10)
        check(
            "task_distance_mae_degradation",
            quantized_mae <= allowed_mae,
            quantized_mae,
            f"<={allowed_mae}",
        )

    negative_count = int((has_track == 0).sum())
    if negative_count > 0:
        float_f1 = float(float_metrics["f1"])
        quantized_f1 = float(quantized_metrics["f1"])
        check(
            "confidence_f1_degradation",
            quantized_f1 >= float_f1 - 0.02,
            quantized_f1,
            f">={float_f1 - 0.02}",
        )
    else:
        check(
            "negative_samples_present",
            False,
            negative_count,
            ">=1",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        output_dir / "predictions.csv",
        rows,
        float_predictions,
        quantized_predictions,
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in checks),
        "mode": "final_acceptance",
        "models": {
            "float": {"path": str(float_model), "sha256": sha256(float_model)},
            "quantized": {
                "path": str(quantized_model),
                "sha256": sha256(quantized_model),
            },
        },
        "manifest": {
            "path": str(manifest),
            "sha256": sha256(manifest),
            "split": args.split,
        },
        "float_metrics": float_metrics,
        "quantized_metrics": quantized_metrics,
        "quantization_drift": {
            "mean_px": float(drift.mean()),
            "median_px": float(np.median(drift)),
            "p90_px": float(np.percentile(drift, 90)),
            "p95_px": float(np.percentile(drift, 95)),
            "max_px": float(drift.max()),
            "confidence_abs_mean": float(
                np.abs(
                    sigmoid(quantized_predictions[:, 2])
                    - sigmoid(float_predictions[:, 2])
                ).mean()
            ),
        },
        "by_video_id": grouped_comparison(
            rows,
            "video_id",
            float_predictions,
            quantized_predictions,
            targets,
            has_track,
            args.confidence_threshold,
        ),
        "by_direction": grouped_comparison(
            rows,
            "direction",
            float_predictions,
            quantized_predictions,
            targets,
            has_track,
            args.confidence_threshold,
        ),
        "by_lighting": grouped_comparison(
            rows,
            "lighting",
            float_predictions,
            quantized_predictions,
            targets,
            has_track,
            args.confidence_threshold,
        ),
        "by_task": grouped_comparison(
            rows,
            "task",
            float_predictions,
            quantized_predictions,
            targets,
            has_track,
            args.confidence_threshold,
        ),
        "checks": checks,
        "warnings": (
            [
                "This split has no negative samples. Confidence precision/F1 and "
                "no-track behavior are not valid final evidence."
            ]
            if negative_count == 0
            else []
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
