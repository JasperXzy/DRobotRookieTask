#!/usr/bin/env python3
"""Record a ROS 2 CompressedImage topic to a video and metadata JSON."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record sensor_msgs/msg/CompressedImage messages to a video. "
            "The Task1 training source must remain /image."
        )
    )
    parser.add_argument(
        "--topic",
        default="/image",
        help="CompressedImage topic (default: /image)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output video path, for example Task1/output/task1_line_001.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback frame rate written to the video container (default: 30)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop this many seconds after the first frame; 0 means Ctrl+C",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec with exactly four characters (default: mp4v)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing video and metadata file",
    )
    return parser.parse_args()


class CompressedImageRecorder(Node):
    def __init__(
        self,
        topic: str,
        output_path: Path,
        fps: float,
        codec: str,
        duration: float,
    ) -> None:
        super().__init__("task1_camera_video_recorder")
        self.topic = topic
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.duration = duration

        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_width = 0
        self.frame_height = 0
        self.frames_received = 0
        self.frames_written = 0
        self.decode_failures = 0
        self.first_frame_monotonic: Optional[float] = None
        self.last_frame_monotonic: Optional[float] = None
        self.started_at_utc: Optional[str] = None
        self.first_ros_stamp_ns: Optional[int] = None
        self.last_ros_stamp_ns: Optional[int] = None
        self.last_frame_id = ""
        self.done = False

        self.subscription = self.create_subscription(
            CompressedImage,
            topic,
            self.on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Waiting for CompressedImage messages on {topic}; output={output_path}"
        )

    def on_image(self, message: CompressedImage) -> None:
        self.frames_received += 1
        now = time.monotonic()

        encoded = np.frombuffer(message.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            self.decode_failures += 1
            if self.decode_failures <= 5:
                self.get_logger().warning("Failed to decode a compressed image frame")
            return

        if self.writer is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height),
            )
            if not self.writer.isOpened():
                self.done = True
                raise RuntimeError(
                    f"OpenCV could not open {self.output_path} with codec {self.codec}"
                )
            self.first_frame_monotonic = now
            self.started_at_utc = datetime.now(timezone.utc).isoformat()
            self.get_logger().info(
                f"Recording {self.frame_width}x{self.frame_height} at container FPS "
                f"{self.fps:.3f}"
            )

        self.writer.write(frame)
        self.frames_written += 1
        self.last_frame_monotonic = now
        self.last_frame_id = message.header.frame_id

        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        if self.first_ros_stamp_ns is None:
            self.first_ros_stamp_ns = stamp_ns
        self.last_ros_stamp_ns = stamp_ns

        if (
            self.duration > 0
            and self.first_frame_monotonic is not None
            and now - self.first_frame_monotonic >= self.duration
        ):
            self.done = True

    def close(self) -> dict:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        elapsed = 0.0
        if (
            self.first_frame_monotonic is not None
            and self.last_frame_monotonic is not None
        ):
            elapsed = max(0.0, self.last_frame_monotonic - self.first_frame_monotonic)

        receive_fps = 0.0
        if elapsed > 0 and self.frames_received > 1:
            receive_fps = (self.frames_received - 1) / elapsed

        ros_elapsed = 0.0
        if (
            self.first_ros_stamp_ns is not None
            and self.last_ros_stamp_ns is not None
            and self.last_ros_stamp_ns >= self.first_ros_stamp_ns
        ):
            ros_elapsed = (self.last_ros_stamp_ns - self.first_ros_stamp_ns) / 1e9

        metadata = {
            "source_topic": self.topic,
            "message_type": "sensor_msgs/msg/CompressedImage",
            "output_video": str(self.output_path.resolve()),
            "codec": self.codec,
            "container_fps": self.fps,
            "width": self.frame_width,
            "height": self.frame_height,
            "frames_received": self.frames_received,
            "frames_written": self.frames_written,
            "decode_failures": self.decode_failures,
            "wall_elapsed_seconds": round(elapsed, 6),
            "ros_stamp_elapsed_seconds": round(ros_elapsed, 6),
            "average_receive_fps": round(receive_fps, 3),
            "frame_id": self.last_frame_id,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return metadata


def validate_args(args: argparse.Namespace) -> Path:
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if args.duration < 0:
        raise ValueError("--duration cannot be negative")
    if len(args.codec) != 4:
        raise ValueError("--codec must contain exactly four characters")

    output_path = Path(args.output).expanduser()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; choose another name or pass --overwrite"
        )

    metadata_path = output_path.with_suffix(".json")
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{metadata_path} already exists; choose another name or pass --overwrite"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def main() -> int:
    args = parse_args()
    try:
        output_path = validate_args(args)
    except (ValueError, FileExistsError) as error:
        print(f"Argument error: {error}", file=sys.stderr)
        return 2

    rclpy.init()
    recorder = CompressedImageRecorder(
        topic=args.topic,
        output_path=output_path,
        fps=args.fps,
        codec=args.codec,
        duration=args.duration,
    )

    exit_code = 0
    try:
        while rclpy.ok() and not recorder.done:
            rclpy.spin_once(recorder, timeout_sec=0.1)
    except KeyboardInterrupt:
        recorder.get_logger().info("Stopping on Ctrl+C")
    except Exception as error:
        recorder.get_logger().error(str(error))
        exit_code = 1
    finally:
        metadata = recorder.close()
        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if metadata["frames_written"] == 0:
        print(
            f"No frames were recorded. Check that {args.topic} exists and is "
            "sensor_msgs/msg/CompressedImage.",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Metadata: {metadata_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
