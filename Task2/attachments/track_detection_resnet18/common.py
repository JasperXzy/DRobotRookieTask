#!/usr/bin/env python3
"""Shared path, configuration, hashing, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TOOLKIT_ROOT = Path(__file__).resolve().parent
TASK2_ROOT = TOOLKIT_ROOT.parents[1]
REPO_ROOT = TASK2_ROOT.parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = resolve_repo_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return data


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a manifest image path relative to Task2/data when needed."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists() or candidate.parts[:1] == ("Task2",):
        return repo_candidate
    return TASK2_ROOT / "data" / candidate


def relative_to_repo(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def relative_to_task2_data(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to((TASK2_ROOT / "data").resolve()).as_posix()
    except ValueError:
        return relative_to_repo(resolved)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
