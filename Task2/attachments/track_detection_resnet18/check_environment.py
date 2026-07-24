#!/usr/bin/env python3
"""Validate the Task2 Conda environment and write a JSON report."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import resolve_repo_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="Task2/output/environment_report.json",
        help="Output JSON report.",
    )
    parser.add_argument(
        "--video",
        help="Optional Task1 MP4 used to verify OpenCV decoding.",
    )
    return parser.parse_args()


def command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return (result.stdout or result.stderr).strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return None


def main() -> int:
    args = parse_args()
    try:
        os_release = platform.freedesktop_os_release()
    except OSError:
        os_release = {}
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "ubuntu": os_release.get("VERSION_ID"),
        "wsl": bool(os.environ.get("WSL_DISTRO_NAME")),
        "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
        "conda": command_version(["conda", "--version"]),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "checks": {},
    }

    try:
        import cv2
        import onnx
        import onnxruntime
        import torch
        import torchvision

        from model import build_model

        report.update(
            {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "opencv": cv2.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": onnxruntime.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
        model = build_model(output_size=3, pretrained=False).eval()
        with torch.no_grad():
            shape = list(model(torch.zeros(1, 3, 224, 224)).shape)
        report["checks"]["resnet18_forward_shape"] = shape
        report["checks"]["imports"] = "passed"
    except Exception as error:  # report every environment failure together
        report["checks"]["imports"] = f"failed: {error}"

    if args.video:
        try:
            import cv2

            video_path = resolve_repo_path(args.video)
            capture = cv2.VideoCapture(str(video_path))
            ok, frame = capture.read()
            capture.release()
            report["checks"]["video_decode"] = bool(ok and frame is not None)
            report["checks"]["video_path"] = str(video_path)
        except Exception as error:
            report["checks"]["video_decode"] = f"failed: {error}"

    output_path = resolve_repo_path(args.output)
    write_json(output_path, report)
    print(f"Wrote {output_path}")

    passed = (
        report.get("conda_env") == "drobot-train"
        and report["checks"].get("imports") == "passed"
        and report["checks"].get("resnet18_forward_shape") == [1, 3]
    )
    if args.video:
        passed = passed and report["checks"].get("video_decode") is True
    if not passed:
        print("Environment check failed; inspect the JSON report.", file=sys.stderr)
        return 1
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
