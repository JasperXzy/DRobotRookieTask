# racing_track_detection

本包用于在 RDK X5 上加载 Task3 生成的 BIN，发布赛道中心点检测结果和
Foxglove `ImageAnnotations`。

## 话题

| 方向 | 话题 | 类型 |
| --- | --- | --- |
| 输入 | `/image_hbmem` | `hbm_img_msgs/msg/HbmMsg1080P` |
| 输入 | `/sign4return` | `std_msgs/msg/Int32` |
| 输出 | `/racing_track_center_detection` | `ai_msgs/msg/PerceptionTargets` |
| 输出 | `/racing_track_annotations` | `foxglove_msgs/msg/ImageAnnotations` |

检测节点按实际相机宽高裁剪底部 `224/480` 区域，模型输出转换到控制器使用的
640×480 坐标。ImageAnnotations 会再按原始图像尺寸缩放，时间戳与对应的
`/image_show` 图像一致。

## 准备模型

在培训仓库根目录执行：

```bash
bash Task3/attachments/racing_track_detection/scripts/stage_model.sh
```

生成模型不会提交 Git。复制本包到板端工作空间时，必须确认
`config/race_track_detection_224x224_nv12.bin` 一同复制。

## 板端依赖与构建

先确认 ImageAnnotations 消息存在：

```bash
source /opt/tros/humble/setup.bash
ros2 interface show foxglove_msgs/msg/ImageAnnotations
```

如果消息不存在：

```bash
sudo apt update
sudo apt install ros-humble-foxglove-msgs
```

构建：

```bash
cd /root/D_Robot/dev_ws
source /opt/tros/humble/setup.bash
colcon build --packages-select racing_track_detection --symlink-install
source install/setup.bash
```

## 启动

```bash
ros2 launch racing_track_detection racing_track_detection.launch.py
```

检查：

```bash
ros2 topic hz /racing_track_center_detection
ros2 topic hz /racing_track_annotations
ros2 topic echo /racing_track_center_detection --once
```

在 Foxglove Image 面板中：

1. 主图像选择 `/image_show`。
2. 在 **Image annotations** 中启用 `/racing_track_annotations`。
3. 启用 **Sync timestamps**。

Foxglove 会直接绘制中心点圆圈和 confidence 文本，不需要发布或重新编码另一条
JPEG 图像流。
