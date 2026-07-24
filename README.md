# DRobotRookieTask

地瓜智慧医疗机器人竞赛新成员培训项目

本项目通过连续的工程任务，帮助新成员熟悉 OriginCarPro 摄像头与底盘驱动、
ROS 2 数据流、Foxglove 远程调试、人工遥操和训练数据采集，并为后续
巡线、避障及自动停车算法开发建立基础。

## 任务入口

| 任务 | 内容 | 状态 | 文档 |
| --- | --- | --- | --- |
| Task1 | 摄像头、Foxglove 与人工遥操数据采集 | 已发布 | [开始 Task1](Task1/Task1.md) |

Task1 配套的 macOS 和 Windows 遥操客户端请从
[GitHub Releases](https://github.com/JasperXzy/DRobotRookieTask/releases)
下载。

## Task1 概览

完成 Task1 后，新成员应能够：

- 启动并说明 `/image`、`/image_hbmem` 和 `/image_show` 三路图像流
- 使用 Foxglove Bridge 查看图像、里程计和速度指令
- 使用 `DRobotTeleop` 通过键盘安全控制小车
- 从 `/image` 录制训练视频
- 人工驾驶小车完成任务一、任务二和任务三，并提交验收材料

具体环境配置、操作命令和验收要求请以
[Task1/Task1.md](Task1/Task1.md) 为准。

## 主要环境

- 地平线 RDK X5
- OriginCar 底盘
- ROS 2 Humble / TROS
- Foxglove Bridge
- Foxglove Desktop
- macOS Apple Silicon 或 Windows x86-64 控制电脑


## License

本项目采用 [Apache License 2.0](LICENSE)，
项目使用的第三方软件和依赖仍分别遵循其各自的许可证。
