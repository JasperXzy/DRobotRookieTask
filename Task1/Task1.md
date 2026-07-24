# Task1：摄像头、Foxglove 与人工遥操数据采集

## 1. 任务定位

Task1 是第一个完整工程任务，完成后，你应当能够独立启动底盘和摄像头、通过 Foxglove 观察 ROS 2 数据并控制小车，以及从 ROS 2 图像话题录制可用于后续 ResNet 巡线训练和 Yolo 避障训练的视频。


## 2. rdk-x5 的话题约定

本任务以 `rdk-x5:/root/D_Robot/dev_ws` 上的 OriginCar 工作空间为准，统一使用以下话题：

| 数据 | 节点 | 话题 | 类型 | 默认状态 |
| --- | --- | --- | --- | --- |
| 摄像头 MJPEG 原始流 | `hobot_usb_cam` | `/image` | `sensor_msgs/msg/CompressedImage` | 开启 |
| NV12 推理流 | `camera_nv12_decoder` | `/image_hbmem` | `hbm_img_msgs/msg/HbmMsg1080P` | 开启 |
| Foxglove 预览流 | `camera_preview_encoder` | `/image_show` | `sensor_msgs/msg/CompressedImage` | 开启 |
| 底盘速度指令 | `origincar_base` | `/cmd_vel` | `geometry_msgs/msg/Twist` | 底盘启动后订阅 |
| 底盘里程计 | `origincar_base` | `/odom` | `nav_msgs/msg/Odometry` | 底盘启动后发布 |



## 3. 学习目标

完成本任务后，你应能：

1. 说明三个摄像头节点分别解决什么问题，以及为什么训练、推理和远程预览使用不同的数据流。
2. 安装 Foxglove，并使用 Foxglove Bridge 连接 ROS 2 Humble/TROS。
3. 在 Foxglove 中查看摄像头、里程计和 `/cmd_vel`。
4. 在 Windows 或 macOS 上直接运行 `DRobotTeleop`，通过 Foxglove
   Bridge 发布 `geometry_msgs/msg/Twist` 并安全控制小车。
5. 使用本任务提供的脚本，从 `/image` 录制训练视频。
6. 人工驾驶小车完成任务一、任务二和任务三的完整路线。
7. 提交可复现的测试记录、视频和问题说明。

## 4. 附件

| 文件 | 用途 |
| --- | --- |
| `attachments/foxglove_task1.json` | Foxglove 布局 |
| `attachments/foxglove_bridge_client/` | 基于 Foxglove Bridge 的跨平台 WASD 遥操客户端、配置和打包说明 |
| `attachments/camera.launch.py` | 板端默认三路摄像头 launch |
| [GitHub Releases](https://github.com/JasperXzy/DRobotRookieTask/releases) | macOS ARM64 与 Windows x86-64 遥操程序下载页 |
| `scripts/record_camera_video.py` | 从 `/image` 录制视频并生成元数据 |
| `templates/acceptance_report.md` | 验收报告模板 |
| `templates/recording_manifest.csv` | 训练视频清单模板 |
| `output/` | 默认视频和元数据输出目录 |

## 5. 前置条件

- 控制电脑已安装 Foxglove Desktop。
- 小车端已安装 `foxglove_bridge`。
- 控制电脑已取得对应平台的 `DRobotTeleop` 可执行程序，或已配置客户端
  Python 环境。
- 录制脚本依赖 `python3-opencv` 和 `python3-numpy`。


小车端安装依赖：

```bash
sudo apt update
sudo apt install ros-humble-foxglove-bridge python3-opencv python3-numpy
```


## 6. 阶段 A：建立基线

### A1. 进入工作空间

```bash
cd ~/D_Robot/dev_ws
source /opt/tros/humble/setup.bash
source install/setup.bash
```

### A2. 确认摄像头设备

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --all
```

`/dev/video0` 是真正的 Video Capture 节点，支持 MJPEG；`/dev/video1`
仅提供 UVC metadata。


### A3. 启动摄像头

将附件复制到板端工作空间并重新编译：

```bash
cp Task1/attachments/camera.launch.py \
  /root/D_Robot/dev_ws/src/origincar/origincar_bringup/launch/
cd /root/D_Robot/dev_ws
source /opt/tros/humble/setup.bash
colcon build --packages-select origincar_bringup
source install/setup.bash
```

启动三路摄像头数据流：

```bash
ros2 launch origincar_bringup camera.launch.py
```

另开终端执行：

```bash
ros2 node list
ros2 topic list -t
ros2 topic hz /image
ros2 topic hz /image_hbmem
ros2 topic hz /image_show
```

目标：

- 三个摄像头节点均存在。
- `/image` 和 `/image_hbmem` 稳定达到 28～30 FPS。
- `/image_show` 稳定达到约 15 FPS，且不会明显拖慢 NV12 推理流。
- 不再启动旧 `image_transport_node` 的 CPU NV12→BGR→JPEG 路径。

### A4. 理解并评估摄像头配置

你需要在验收报告中解释：

- 为什么 USB 摄像头直接输出 MJPEG。
- 为什么 BPU 推理使用 NV12 shared memory。
- 为什么远程预览单独重新编码，并使用较低 JPEG 质量。
- 为什么训练视频必须来自 `/image`，而不是低质量 `/image_show`。
- `640x480 @ 30 FPS`、`1280x720 @ 30 FPS`、`1920x1080 @ 30 FPS` 对网络、CPU、VPU 和存储的影响。

## 7. 阶段 B：Foxglove Bridge

### B1. 启动 Bridge

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

Foxglove Bridge 默认监听端口通常为 `8765`。在控制电脑的 Foxglove 中选择 **Foxglove WebSocket**，连接：

```text
ws://<小车IP>:8765
```


### B2. 准备独立遥操客户端

打开
[JasperXzy/DRobotRookieTask Releases](https://github.com/JasperXzy/DRobotRookieTask/releases)，
进入最新版本，在 **Assets** 中下载与控制电脑匹配的压缩包：

- Apple Silicon macOS：`DRobotTeleop-macos-arm64.zip`。
- 64 位 Windows：`DRobotTeleop-windows-x86_64.zip`。
- 可选下载 `ARCHIVE_SHA256SUMS`，用于核对压缩包完整性。

不要下载 GitHub 自动生成的 `Source code (zip)`；它是仓库源码快照，不是可直接
运行的遥操客户端。

下载后解压，得到对应平台目录：

```text
macOS:
macos-arm64/
├── DRobotTeleop
├── config.yaml
└── map.png

Windows:
windows-x86_64/
├── DRobotTeleop.exe
├── config.yaml
└── map.png
```

先修改同目录的 `config.yaml`：

```yaml
bridge_url: ws://<小车IP>:8765
```


启动方式：

- macOS：双击 `DRobotTeleop`。
- Windows：双击 `DRobotTeleop.exe`。

打包程序退出前默认等待回车，避免双击启动后的错误信息一闪而过。

源码和维护者打包说明保留在
`attachments/foxglove_bridge_client/`，不属于新成员安装步骤。

#### macOS 首次授权

macOS 必须为当前构建出的 `DRobotTeleop` 同时开启：

1. `系统设置 → 隐私与安全性 → 辅助功能`。
2. `系统设置 → 隐私与安全性 → 输入监控`。


遥操期间终端会隐藏键盘字符，所以按 `R` 或 WASD 时不显示字母，这是预期
行为，不是在请求密码，也不需要按回车。若按 `R` 完全没有反应，检查
`Terminal → Secure Keyboard Entry`，确保该选项没有勾选；安全键盘输入会
阻止 `pynput` 取得按键。

### B3. 导入布局

在 Foxglove 的 Layouts 菜单选择 **Import from file**，导入：

```text
attachments/foxglove_task1.json
```

导入后检查：

- Image 面板选择 `/image_show`。
- Raw Messages 面板选择 `/odom`。

### B4. 启动遥操

确认 rdk-x5 上的底盘和 Foxglove Bridge 已启动，然后直接双击对应平台的
可执行程序。看到“已连接 Foxglove Bridge，`clientPublish/json` 可用”后，
说明客户端连接成功。

客户端启动时处于暂停状态，不会立即驱动车辆；按 `R` 后才开始以 20 Hz
发布。控制规则与
`references/foxglove/src/bridge_client.py` 保持一致：

- `W` 前进，`S` 后退。
- `W+A`/`W+D` 前进左转/右转。
- `S+D`/`S+A` 后退左转/右转。
- 单按 `A/D` 只发布 `angular.z`；Ackermann 小车静止时舵机不动作。
- `↑/↓` 增加/减小线速度；`←/→` 减小/增加转向值，方向键本身不驾驶。
- `P` 暂停并停车，`R` 恢复，`Esc` 停车并退出。
- 没有按下 WASD 时持续发布零速度。
- 持续按住 WASD 不应在终端产生连续字符输出。


## 8. 阶段 C：遥操安全测试

### C1. 启动底盘

```bash
ros2 launch origincar_base origincar_bringup.launch.py
```

### C2. 架空测试

依次完成：

1. 按 `P`，确认客户端发送停止且四个轮子不转。
2. 低速前进，确认方向正确。
3. 低速后退，确认方向正确。
4. 使用 `W+A`/`W+D` 完成前进转向。
5. 单独按 `A/D`，确认消息在发布，但车辆和舵机不动作。
6. 使用方向键调整数值，确认车辆不会仅因方向键而移动。
7. 松开控制后确认立即发布零速度。
8. 按 `P` 或 `Esc` 正常停止客户端，确认退出前发送零速度。

STM32 的 Ackermann 固件使用 `Vx/Vz` 计算前轮舵角；当
`linear.x=0, angular.z≠0` 时会把舵角置零。因此实际转向必须使用
`W+A`、`W+D`、`S+A` 或 `S+D` 组合键，base 驱动不会把单独 `A/D`
自动改造成带速度的命令。

### C3. 落地低速测试

- 初始线速度不得超过 `0.3 m/s`。
- 先完成直线、停车、S 弯，再进入完整赛道。

## 9. 阶段 D：录制训练视频

### D1. 启动录制

在本培训仓库根目录执行：

```bash
source /opt/tros/humble/setup.bash
python3 Task1/scripts/record_camera_video.py \
  --topic /image \
  --output Task1/output/task1_line_001.mp4 \
  --fps 30
```

按 `Ctrl+C` 停止。也可以指定时长：

```bash
python3 Task1/scripts/record_camera_video.py \
  --topic /image \
  --output Task1/output/task1_line_001.mp4 \
  --fps 30 \
  --duration 180
```

脚本会在视频旁生成同名 `.json` 元数据文件，记录图像尺寸、帧数、接收帧率、录制时间和源话题。

### D2. 视频采集要求

每次录制前清洁镜头并固定相机安装角度。至少采集：

| 场景 | 最低要求 |
| --- | --- |
| 任务一完整路线 | 至少 3 遍 |
| 任务二完整路线 | 顺时针逆时针各至少 2 遍 |
| 任务三完整路线 | 至少 3 遍 |
| 光照 | 正常、偏暗、局部强光至少两类 |
| 驾驶偏差 | 包含居中、轻微左偏、轻微右偏 |
| 障碍物 | 覆盖不同距离和横向位置 |
| 停车区 | 包含正常停车和至少一次过早/过晚停车样本 |


每段视频都要登记到 `templates/recording_manifest.csv` 的副本中。

## 10. 阶段 E：完整任务验收

人工驾驶依次完成：

### 任务一

- 完成巡线区域。
- 正确绕过障碍物。
- 到达二维码区域并保证二维码清晰入镜。

### 任务二

- 使用远程遥操完成规定路线。
- 全程视频连续，无长时间卡顿或黑屏。
- 轻微网络抖动后能够人工停车并恢复；突然断联不作为自动停车验收项。

### 任务三

- 完成巡线和避障路线。
- 进入停车区域。
- 在目标范围内人工停车，并保留足够的停车牌/停车区画面。

每个任务至少保留：

- 一段完整车载视频。
- 问题与改进记录。

## 11. 验收标准

| 类别 | 验收项 | 通过条件 |
| --- | --- | --- |
| 环境 | Foxglove Bridge | 可通过 `ws://<IP>:8765` 稳定连接 |
| 摄像头 | 原始训练流 | `/image` 类型正确，28～30 FPS |
| 摄像头 | NV12 推理流 | `/image_hbmem` 类型正确，28～30 FPS |
| 摄像头 | 远程预览流 | `/image_show` 类型正确，约 15 FPS |
| 遥操 | 控制能力 | WASD 完成前后与组合转向；方向键正确调整速度/转向值 |
| 遥操 | 终端显示 | 持续按住 WASD 不产生连续字符，状态和错误信息仍可见 |
| 安全 | 松手停车 | 松开控制立即发送零速度 |
| 安全 | 正常退出停车 | `P`、`Esc` 或正常退出客户端时发送零速度 |
| 录制 | 数据源 | 脚本订阅 `/image` |
| 录制 | 视频有效 | 视频可播放、无明显丢帧、元数据完整 |
| 场地 | 完整任务 | 人工完成任务一、二、三并提交证据 |


## 12. 快速检查

摄像头、底盘和 Bridge 启动后，在 rdk-x5 上执行以下只读命令：

```bash
ros2 node list
ros2 topic list -t
ros2 topic info /cmd_vel
ros2 topic info /odom
ros2 topic hz /image
ros2 topic hz /image_hbmem
ros2 topic hz /image_show
```

## 13. 提交物

成员提交以下内容：

```text
Task1_submission_<组别>/
├── acceptance_report.md
├── recording_manifest.csv
```


## 14. 参考资料

- `attachments/foxglove_bridge_client/README.md`
- `rdk-x5:/root/D_Robot/dev_ws/src/origincar/origincar_bringup/launch/usb_websocket_display.launch.py`
- `rdk-x5:/root/D_Robot/dev_ws/src/origincar/origincar_base/launch/base_serial.launch.py`
- `rdk-x5:/root/D_Robot/dev_ws/src/origincar/origincar_base/src/origincar_base.cpp`
- [Foxglove：ROS 2 实时连接](https://docs.foxglove.dev/docs/getting-started/frameworks/ros2)
- [Foxglove：布局导入与导出](https://docs.foxglove.dev/docs/visualization/layouts)
