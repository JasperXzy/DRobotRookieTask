#!/usr/bin/env python3
"""ResNet18 model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.models import ResNet18_Weights, resnet18


def build_model(output_size: int = 3, pretrained: bool = True) -> torch.nn.Module:
    if output_size not in {2, 3}:
        raise ValueError("output_size must be 2 or 3")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, output_size)
    return model


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        output_size = int(checkpoint.get("output_size", 3))
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint
    else:
        state_dict = checkpoint
        output_size = int(state_dict["fc.weight"].shape[0])
        metadata = {"output_size": output_size}

    model = build_model(output_size=output_size, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata
