#!/usr/bin/env python3
"""Validate the host, OpenExplorer tools, ONNX contract, and PTQ config."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_INPUT_NAME = "input"
EXPECTED_INPUT_SHAPE = [1, 3, 224, 224]
EXPECTED_OUTPUT_NAME = "output"
EXPECTED_OUTPUT_SHAPE = [1, 3]
EXPECTED_OPSET = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-missing-tools",
        action="store_true",
        help="Report missing OpenExplorer commands without returning a failure.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_version(command: str) -> dict[str, Any]:
    executable = shutil.which(command)
    result: dict[str, Any] = {"found": executable is not None, "path": executable}
    if executable is None:
        return result

    for version_args in (("--version",), ("-v",), ("--help",)):
        try:
            completed = subprocess.run(
                [executable, *version_args],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = (completed.stdout or completed.stderr).strip()
        if text:
            result["version_text"] = text.splitlines()[0][:500]
            result["returncode"] = completed.returncode
            break
    return result


def import_version(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exact SDK import failures vary.
        return {"found": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "found": True,
        "version": getattr(module, "__version__", None),
        "path": getattr(module, "__file__", None),
    }


def tensor_shape(value_info: Any) -> list[int | str | None]:
    dims: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


def inspect_onnx(path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return {"path": str(path), "exists": False}, [f"ONNX model not found: {path}"]

    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = model.graph
    inputs = [{"name": item.name, "shape": tensor_shape(item)} for item in graph.input]
    outputs = [{"name": item.name, "shape": tensor_shape(item)} for item in graph.output]
    opsets = {item.domain or "ai.onnx": int(item.version) for item in model.opset_import}

    if len(inputs) != 1:
        errors.append(f"Expected exactly one input, found {len(inputs)}.")
    elif inputs[0] != {"name": EXPECTED_INPUT_NAME, "shape": EXPECTED_INPUT_SHAPE}:
        errors.append(
            f"Input contract mismatch: expected {EXPECTED_INPUT_NAME} "
            f"{EXPECTED_INPUT_SHAPE}, found {inputs[0]}."
        )

    if len(outputs) != 1:
        errors.append(f"Expected exactly one output, found {len(outputs)}.")
    elif outputs[0] != {"name": EXPECTED_OUTPUT_NAME, "shape": EXPECTED_OUTPUT_SHAPE}:
        errors.append(
            f"Output contract mismatch: expected {EXPECTED_OUTPUT_NAME} "
            f"{EXPECTED_OUTPUT_SHAPE}, found {outputs[0]}."
        )

    if opsets.get("ai.onnx") != EXPECTED_OPSET:
        errors.append(
            f"ONNX opset mismatch: expected {EXPECTED_OPSET}, "
            f"found {opsets.get('ai.onnx')}."
        )

    return (
        {
            "path": str(path.resolve()),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "inputs": inputs,
            "outputs": outputs,
            "opsets": opsets,
            "node_count": len(graph.node),
        },
        errors,
    )


def inspect_config(path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return {"path": str(path), "exists": False}, [f"Config not found: {path}"]

    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required = {
        "model_parameters": {
            "march": "bayes-e",
            "onnx_model": "/workspace/Task2/output/line_follower_resnet18.onnx",
        },
        "input_parameters": {
            "input_name": EXPECTED_INPUT_NAME,
            "input_shape": "1x3x224x224",
            "input_type_rt": "nv12",
            "input_type_train": "rgb",
            "input_layout_train": "NCHW",
            "norm_type": "data_mean_and_scale",
        },
        "calibration_parameters": {
            "cal_data_type": "float32",
            "preprocess_on": False,
        },
    }

    for section, expected_values in required.items():
        actual_section = config.get(section)
        if not isinstance(actual_section, dict):
            errors.append(f"Config section is missing or invalid: {section}")
            continue
        for key, expected in expected_values.items():
            actual = actual_section.get(key)
            if actual != expected:
                errors.append(
                    f"Config mismatch at {section}.{key}: "
                    f"expected {expected!r}, found {actual!r}."
                )

    return {
        "path": str(path.resolve()),
        "exists": True,
        "sha256": file_sha256(path),
        "config": config,
    }, errors


def main() -> int:
    args = parse_args()
    errors: list[str] = []

    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        errors.append(
            "OpenExplorer PTQ must run in the Linux x86_64 Docker container; "
            f"current platform is {platform.system()} {platform.machine()}."
        )

    modules = {
        name: import_version(name)
        for name in ("onnx", "onnxruntime", "yaml", "horizon_tc_ui")
    }
    for required_module in ("onnx", "onnxruntime", "yaml", "horizon_tc_ui"):
        if not modules[required_module]["found"]:
            errors.append(f"Required Python module is unavailable: {required_module}")

    commands = {
        name: command_version(name)
        for name in ("hb_mapper", "hb_model_info", "hb_perf", "hb_verifier")
    }
    for required_command in ("hb_mapper", "hb_model_info", "hb_perf", "hb_verifier"):
        if not commands[required_command]["found"] and not args.allow_missing_tools:
            errors.append(f"Required OpenExplorer command is unavailable: {required_command}")

    onnx_report: dict[str, Any]
    config_report: dict[str, Any]
    try:
        onnx_report, onnx_errors = inspect_onnx(args.onnx.expanduser().resolve())
        errors.extend(onnx_errors)
    except Exception as exc:
        onnx_report = {"path": str(args.onnx), "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"Failed to inspect ONNX model: {exc}")

    try:
        config_report, config_errors = inspect_config(args.config.expanduser().resolve())
        errors.extend(config_errors)
    except Exception as exc:
        config_report = {"path": str(args.config), "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"Failed to inspect PTQ config: {exc}")

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "python": sys.version,
        },
        "modules": modules,
        "commands": commands,
        "onnx": onnx_report,
        "config": config_report,
        "errors": errors,
    }

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
