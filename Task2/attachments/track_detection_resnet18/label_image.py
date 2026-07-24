#!/usr/bin/env python3
"""Interactively label one track target point in each 640x224 ROI image.

Current label format:

    x_roi y_roi state

state=1: positive sample with raw ROI pixel coordinates
state=0: negative sample, stored as "nan nan 0"
state=-1: rejected sample, stored as "nan nan -1"

The script can also display legacy two-value normalized labels so existing
annotations remain readable.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
POSITIVE = 1
NEGATIVE = 0
REJECTED = -1


@dataclass
class Label:
    x: float
    y: float
    state: int
    legacy: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label one track-center point per 640x224 ROI image."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing ROI images.",
    )
    parser.add_argument(
        "--label-dir",
        required=True,
        help="Directory where same-stem .txt labels are stored.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=2.0,
        help="Display scale only; saved coordinates remain in ROI pixels.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based image index to open first.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Start from the first image without a label.",
    )
    parser.add_argument(
        "--expected-width",
        type=int,
        default=640,
        help="Required ROI width.",
    )
    parser.add_argument(
        "--expected-height",
        type=int,
        default=224,
        help="Required ROI height.",
    )
    parser.add_argument(
        "--band-min",
        type=int,
        default=80,
        help="Top row of the displayed look-ahead guide band.",
    )
    parser.add_argument(
        "--band-max",
        type=int,
        default=144,
        help="Bottom row of the displayed look-ahead guide band.",
    )
    return parser.parse_args()


def read_label(path: Path, width: int, height: int) -> Optional[Label]:
    if not path.exists():
        return None

    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) == 3:
        x = float(fields[0])
        y = float(fields[1])
        state = int(fields[2])
        if state not in {POSITIVE, NEGATIVE, REJECTED}:
            raise ValueError(f"{path}: unsupported state {state}")
        if state == POSITIVE:
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"{path}: positive label requires finite coordinates")
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"{path}: point ({x}, {y}) is outside {width}x{height}"
                )
        return Label(x=x, y=y, state=state)

    if len(fields) == 2:
        # Compatibility with the earlier two-field [0, 1] label format.
        x_norm = float(fields[0])
        y_norm = float(fields[1])
        if not (0 <= x_norm <= 1 and 0 <= y_norm <= 1):
            raise ValueError(
                f"{path}: legacy two-field coordinates must be in [0, 1]"
            )
        return Label(
            x=min(width - 1.0, x_norm * width),
            y=min(height - 1.0, y_norm * height),
            state=POSITIVE,
            legacy=True,
        )

    raise ValueError(f"{path}: expected 2 legacy fields or 3 current fields")


def write_label(path: Path, label: Label) -> None:
    if label.state == POSITIVE:
        content = f"{label.x:.3f} {label.y:.3f} 1\n"
    elif label.state == NEGATIVE:
        content = "nan nan 0\n"
    elif label.state == REJECTED:
        content = "nan nan -1\n"
    else:
        raise ValueError(f"unsupported label state {label.state}")
    path.write_text(content, encoding="utf-8")


def state_name(label: Optional[Label]) -> str:
    if label is None:
        return "UNLABELED"
    if label.state == POSITIVE:
        return "POSITIVE (legacy)" if label.legacy else "POSITIVE"
    if label.state == NEGATIVE:
        return "NEGATIVE"
    return "REJECTED"


class ImageLabeler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.image_dir = Path(args.image_dir).expanduser()
        self.label_dir = Path(args.label_dir).expanduser()
        self.zoom = args.zoom
        self.expected_width = args.expected_width
        self.expected_height = args.expected_height
        self.band_min = args.band_min
        self.band_max = args.band_max

        if self.zoom <= 0:
            raise ValueError("--zoom must be greater than zero")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"image directory does not exist: {self.image_dir}")
        if not 0 <= self.band_min < self.band_max < self.expected_height:
            raise ValueError("look-ahead band must be inside the expected ROI")

        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"no supported images found in {self.image_dir}")

        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.index = max(0, min(args.start_index, len(self.image_paths) - 1))
        if args.resume:
            self.index = self.first_unlabeled_index(default=self.index)

        self.window_name = "Task2 Track Point Labeler"
        self.image = None
        self.display_width = 0
        self.display_height = 0

    def label_path(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def first_unlabeled_index(self, default: int) -> int:
        for index, image_path in enumerate(self.image_paths):
            if not self.label_path(image_path).exists():
                return index
        return default

    def current_image_path(self) -> Path:
        return self.image_paths[self.index]

    def load_current(self):
        image_path = self.current_image_path()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read {image_path}")
        height, width = image.shape[:2]
        if (width, height) != (self.expected_width, self.expected_height):
            raise ValueError(
                f"{image_path}: expected {self.expected_width}x"
                f"{self.expected_height}, got {width}x{height}"
            )
        self.image = image
        return image

    def render(self) -> None:
        image = self.load_current()
        height, width = image.shape[:2]
        label_path = self.label_path(self.current_image_path())
        label = read_label(label_path, width, height)

        canvas = image.copy()
        cv2.line(
            canvas,
            (0, self.band_min),
            (width - 1, self.band_min),
            (0, 255, 255),
            1,
        )
        cv2.line(
            canvas,
            (0, self.band_max),
            (width - 1, self.band_max),
            (0, 255, 255),
            1,
        )

        if label is not None and label.state == POSITIVE:
            cv2.circle(
                canvas,
                (round(label.x), round(label.y)),
                6,
                (0, 0, 255),
                -1,
            )

        status = (
            f"{self.index + 1}/{len(self.image_paths)} "
            f"{self.current_image_path().name} {state_name(label)}"
        )
        cv2.rectangle(canvas, (0, 0), (width - 1, 24), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            status,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        self.display_width = round(width * self.zoom)
        self.display_height = round(height * self.zoom)
        display = cv2.resize(
            canvas,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow(self.window_name, display)
        print(status)
        if label is not None and label.state == POSITIVE:
            print(f"  point=({label.x:.3f}, {label.y:.3f})")

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONUP or self.image is None:
            return
        height, width = self.image.shape[:2]
        x_roi = min(width - 1.0, max(0.0, x * width / self.display_width))
        y_roi = min(height - 1.0, max(0.0, y * height / self.display_height))
        label = Label(x=x_roi, y=y_roi, state=POSITIVE)
        write_label(self.label_path(self.current_image_path()), label)
        print(f"Saved positive: ({x_roi:.3f}, {y_roi:.3f})")
        self.render()

    def move(self, delta: int) -> None:
        self.index = max(0, min(self.index + delta, len(self.image_paths) - 1))
        self.render()

    def save_state(self, state: int) -> None:
        label = Label(x=float("nan"), y=float("nan"), state=state)
        write_label(self.label_path(self.current_image_path()), label)
        print(f"Saved {state_name(label).lower()}")
        self.render()

    @staticmethod
    def print_help() -> None:
        print(
            "\nControls:\n"
            "  Left click  save positive point\n"
            "  N           save negative (nan nan 0)\n"
            "  X           save rejected (nan nan -1)\n"
            "  A / D       previous / next image\n"
            "  Z / C       jump backward / forward 5 images\n"
            "  H           show this help\n"
            "  Q / Esc     quit\n"
        )

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.print_help()
        self.render()

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key == ord("a"):
                self.move(-1)
            elif key == ord("d"):
                self.move(1)
            elif key == ord("z"):
                self.move(-5)
            elif key == ord("c"):
                self.move(5)
            elif key == ord("n"):
                self.save_state(NEGATIVE)
            elif key == ord("x"):
                self.save_state(REJECTED)
            elif key == ord("h"):
                self.print_help()
            else:
                print("Unknown key; press H for help")

        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    try:
        ImageLabeler(args).run()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
