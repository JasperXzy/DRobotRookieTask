#!/usr/bin/env python3
"""Export a Task2 checkpoint to fixed-shape ONNX and run onnx.checker."""

from __future__ import annotations

import argparse
import json
import sys

import onnx
import torch

from common import load_yaml, resolve_repo_path, sha256_file, write_json
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default=(
            "Task2/attachments/track_detection_resnet18/"
            "configs/train_resnet18_xyv.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        default="Task2/output/line_follower_resnet18.onnx",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    export_config = config["export"]
    checkpoint_path = resolve_repo_path(args.checkpoint)
    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, metadata = load_checkpoint(checkpoint_path, device="cpu")
    output_size = int(metadata.get("output_size", 3))
    model.eval()

    example = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(output_path),
        export_params=True,
        opset_version=int(export_config["onnx_opset"]),
        do_constant_folding=True,
        input_names=[str(export_config["input_name"])],
        output_names=[str(export_config["output_name"])],
        dynamic_axes=None,
        dynamo=False,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx": str(output_path),
        "onnx_sha256": sha256_file(output_path),
        "input_name": str(export_config["input_name"]),
        "input_shape": [1, 3, 224, 224],
        "output_name": str(export_config["output_name"]),
        "output_shape": [1, output_size],
        "opset": int(export_config["onnx_opset"]),
        "dynamic_axes": False,
        "onnx_checker": "passed",
    }
    write_json(output_path.with_suffix(".json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
