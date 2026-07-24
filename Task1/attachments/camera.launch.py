"""Default camera pipeline for the OriginCar workspace on RDK X5.

Publishes:
  /image        sensor_msgs/msg/CompressedImage (camera MJPEG)
  /image_hbmem  hbm_img_msgs/msg/HbmMsg1080P (NV12 shared memory)
  /image_show   sensor_msgs/msg/CompressedImage (Foxglove preview JPEG)
"""

import os

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    calibration_file = os.path.join(
        get_package_prefix("hobot_usb_cam"),
        "lib/hobot_usb_cam/config/usb_camera_calibration.yaml",
    )

    device = LaunchConfiguration("device")
    image_width = LaunchConfiguration("image_width")
    image_height = LaunchConfiguration("image_height")
    camera_fps = LaunchConfiguration("camera_fps")
    preview_fps = LaunchConfiguration("preview_fps")
    preview_jpeg_quality = LaunchConfiguration("preview_jpeg_quality")

    shm_environment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_shm"),
                "launch/hobot_shm.launch.py",
            )
        )
    )

    camera_source = Node(
        package="hobot_usb_cam",
        executable="hobot_usb_cam",
        name="hobot_usb_cam",
        output="screen",
        parameters=[
            {
                "camera_calibration_file_path": calibration_file,
                "frame_id": "default_usb_cam",
                "framerate": camera_fps,
                "image_height": image_height,
                "image_width": image_width,
                "io_method": "mmap",
                "pixel_format": "mjpeg",
                "video_device": device,
                "zero_copy": False,
            }
        ],
        arguments=["--ros-args", "--log-level", "warn"],
    )

    nv12_decoder = Node(
        package="hobot_codec",
        executable="hobot_codec_republish",
        name="camera_nv12_decoder",
        output="screen",
        parameters=[
            {
                "channel": 0,
                "in_mode": "ros",
                "in_format": "jpeg",
                "out_mode": "shared_mem",
                "out_format": "nv12",
                "sub_topic": "/image",
                "pub_topic": "/image_hbmem",
                "input_framerate": camera_fps,
                "output_framerate": -1,
                "dump_output": False,
            }
        ],
        arguments=["--ros-args", "--log-level", "warn"],
    )

    preview_encoder = Node(
        package="hobot_codec",
        executable="hobot_codec_republish",
        name="camera_preview_encoder",
        output="screen",
        parameters=[
            {
                "channel": 1,
                "in_mode": "shared_mem",
                "in_format": "nv12",
                "out_mode": "ros",
                "out_format": "jpeg",
                "sub_topic": "/image_hbmem",
                "pub_topic": "/image_show",
                "jpg_quality": preview_jpeg_quality,
                "input_framerate": camera_fps,
                "output_framerate": preview_fps,
                "dump_output": False,
            }
        ],
        arguments=["--ros-args", "--log-level", "warn"],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "device",
                default_value="/dev/video0",
                description="USB camera device",
            ),
            DeclareLaunchArgument(
                "image_width",
                default_value="1280",
                description="Camera image width",
            ),
            DeclareLaunchArgument(
                "image_height",
                default_value="720",
                description="Camera image height",
            ),
            DeclareLaunchArgument(
                "camera_fps",
                default_value="30",
                description="Camera and NV12 pipeline frame rate",
            ),
            DeclareLaunchArgument(
                "preview_fps",
                default_value="15",
                description="Foxglove preview frame rate",
            ),
            DeclareLaunchArgument(
                "preview_jpeg_quality",
                default_value="15.0",
                description="Foxglove preview JPEG quality, from 0 to 100",
            ),
            shm_environment,
            camera_source,
            nv12_decoder,
            preview_encoder,
        ]
    )
