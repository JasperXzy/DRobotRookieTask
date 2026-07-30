"""Launch the X5 BPU racing-track detector."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("racing_track_detection")
    default_model = os.path.join(
        package_share,
        "config",
        "race_track_detection_224x224_nv12.bin",
    )

    model_path = LaunchConfiguration("model_path")
    sub_img_topic = LaunchConfiguration("sub_img_topic")
    output_topic = LaunchConfiguration("output_topic")
    sign_topic = LaunchConfiguration("sign_topic")
    camera_frame_id = LaunchConfiguration("camera_frame_id")
    annotations_topic = LaunchConfiguration("annotations_topic")
    annotation_confidence_threshold = LaunchConfiguration(
        "annotation_confidence_threshold"
    )
    bpu_core_id = LaunchConfiguration("bpu_core_id")
    log_level = LaunchConfiguration("log_level")

    detector = Node(
        package="racing_track_detection",
        executable="racing_track_detection",
        name="racing_track_detection",
        output="screen",
        parameters=[
            {
                "model_path": model_path,
                "sub_img_topic": sub_img_topic,
                "output_topic": output_topic,
                "sign_topic": sign_topic,
                "camera_frame_id": camera_frame_id,
                "annotations_topic": annotations_topic,
                "annotation_confidence_threshold": ParameterValue(
                    annotation_confidence_threshold,
                    value_type=float,
                ),
                "bpu_core_id": ParameterValue(bpu_core_id, value_type=int),
            }
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path", default_value=default_model),
            DeclareLaunchArgument("sub_img_topic", default_value="/image_hbmem"),
            DeclareLaunchArgument(
                "output_topic",
                default_value="/racing_track_center_detection",
            ),
            DeclareLaunchArgument("sign_topic", default_value="/sign4return"),
            DeclareLaunchArgument(
                "camera_frame_id",
                default_value="default_usb_cam",
            ),
            DeclareLaunchArgument(
                "annotations_topic",
                default_value="/racing_track_annotations",
            ),
            DeclareLaunchArgument(
                "annotation_confidence_threshold",
                default_value="0.5",
            ),
            DeclareLaunchArgument("bpu_core_id", default_value="1"),
            DeclareLaunchArgument("log_level", default_value="warn"),
            detector,
        ]
    )
