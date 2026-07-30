# Task3 验收报告

## 基本信息

- 组别：
- 日期：
- WSL 发行版：
- OpenExplorer 版本：
- RDK X5 编号：
- 开发板系统版本：

## 输入冻结

| 项目 | 路径或版本 | SHA256 / 实测值 |
| --- | --- | --- |
| 浮点 ONNX |  |  |
| Task2 数据集 manifest |  |  |
| Task2 train config |  |  |
| Task2 test metrics |  |  |
| PTQ YAML |  |  |
| OpenExplorer Docker image ID |  |  |

## 核心验收

| 类别 | 验收项 | 实测值或证据 | 结果 |
| --- | --- | --- | --- |
| 环境 | Docker Engine 运行于 WSL2 Linux x86-64 |  |  |
| 环境 | OpenExplorer v1.2.8 CPU 容器可复现进入 |  |  |
| 模型 | ONNX 输入 `[1,3,224,224]`、输出 `[1,3]`、opset 11 |  |  |
| 模型 | `hb_mapper checker` 无阻断错误 |  |  |
| 数据 | 校准集只来自 train/val，且 manifest 可追溯 |  |  |
| 数据 | 64 个 RGB CHW float32 文件均为 602112 bytes |  |  |
| PTQ | `hb_mapper makertbin` 生成量化 ONNX 与 BIN |  |  |
| PTQ | 量化坐标平均漂移 ≤ 2 px |  |  |
| PTQ | 量化坐标 P95 漂移 ≤ 5 px |  |  |
| PTQ | 任务距离 MAE 增量不超过 `max(2 px, 10%)` |  |  |
| PTQ | 有负样本时置信度 F1 下降 ≤ 0.02 |  |  |
| PTQ | `hb_verifier` 的量化 ONNX/BIN 一致性通过 |  |  |
| 部署 | 板端 `hrt_model_exec model_info/perf` 可加载 BIN |  |  |
| ROS 2 | `/image_hbmem` 到检测输出稳定运行 |  |  |
| ROS 2 | 输出坐标符合 640×480、左上原点、y 向下契约 |  |  |
| 性能 | 单线程板端模型 latency 和 ROS 2 实测 FPS 达标 |  |  |
| Foxglove | `/racing_track_annotations` 点位和置信度正确 |  |  |
| 安全 | 静止验证完成后才进行低速闭环测试 |  |  |

结果填写：`通过`、`不通过` 或 `不适用`。

## 校准数据

| 项目 | 实测值 |
| --- | --- |
| 候选样本数 |  |
| 最终样本数 |  |
| train / val 数量 |  |
| 正 / 负 / 未标注数量 |  |
| 视频数 |  |
| 方向与光照覆盖 |  |
| `calibration_manifest.csv` SHA256 |  |

## 量化结果

| 指标 | 浮点 ONNX | 量化 ONNX | 变化 |
| --- | ---: | ---: | ---: |
| x MAE / px |  |  |  |
| y MAE / px |  |  |  |
| 二维距离 MAE / px |  |  |  |
| 二维距离 P95 / px |  |  |  |
| 20 px 内命中率 |  |  |  |
| 置信度 Precision |  |  |  |
| 置信度 Recall |  |  |  |
| 置信度 F1 |  |  |  |

### 量化漂移

| 指标 | 实测值 | 门槛 |
| --- | ---: | ---: |
| 平均坐标漂移 / px |  | ≤ 2 |
| P95 坐标漂移 / px |  | ≤ 5 |
| 最大坐标漂移 / px |  | 记录，不单独判定 |
| 平均置信度绝对差 |  | 记录 |

## BIN 与板端性能

| 项目 | 实测值 |
| --- | --- |
| BIN 路径 |  |
| BIN SHA256 |  |
| BPU march |  |
| 输入类型 |  |
| 输出类型与形状 |  |
| 静态估算 latency |  |
| `hrt_model_exec` 单线程 latency |  |
| `hrt_model_exec` FPS |  |
| ROS 2 检测话题 FPS |  |
| CPU / BPU / 内存观察 |  |

## Foxglove 与闭环观察

- 直道点位：
- 左弯点位：
- 右弯点位：
- 暗光点位：
- 无赛道置信度：
- 检测时间戳与图像时间戳：
- 低速闭环最大速度：
- 异常时的停车措施：

## 问题排查记录

至少记录一次实际故障或主动检查。

| 现象或风险 | 定位证据 | 原因 | 修复 | 防复发措施 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## 学习总结与思考

### 1. 环境与边界

- 为什么 OpenExplorer 在 WSL x86-64 Docker 中运行，而不是直接在 Windows 中运行？
- 为什么使用 WSL 原生 Docker 时应避免同时启用 Docker Desktop 的同一发行版集成？
- Docker 镜像、SDK 发布包和板端运行时分别承担什么职责？

### 2. PTQ

- 校准集为什么不能使用 test？为什么不需要标签但需要代表性？
- `input_type_train=rgb` 与 `input_type_rt=nv12` 分别描述什么？
- 为什么校准二进制是 RGB float32 `[0,255]`，不是已经归一化的 Tensor？
- 量化 ONNX 与浮点 ONNX 的任务指标和逐样本漂移分别回答什么问题？

### 3. 部署

- 为什么模型输入是 224×224，但 ROS 2 输出使用 640×480 坐标？
- 为什么 y 坐标必须加 256，且不能上下翻转？
- 为什么低置信度帧仍应发布给控制器？
- 静态 `hb_perf`、板端 `hrt_model_exec` 和 ROS 2 话题 FPS 有什么区别？

### 4. 数据门禁

- 若训练集没有负样本，`track_logit` 学到了什么、没学到什么？
- 为什么缺少负样本时不能执行最终量化验收？
- 下一轮应补采哪些无赛道、遮挡、强光和偏航场景？

## 提交清单

- [ ] `acceptance_report.md`
- [ ] `environment_report.json`
- [ ] `calibration_manifest.csv`
- [ ] `calibration_report.json`
- [ ] `resnet18_xyv.yaml`
- [ ] `hb_mapper_checker.log`
- [ ] `hb_mapper_makertbin.log`
- [ ] `hb_model_info.log`
- [ ] `hb_perf.log`
- [ ] `hb_verifier.log`
- [ ] `quantized_evaluation/metrics.json`
- [ ] `quantized_evaluation/predictions.csv`
- [ ] `race_track_detection_224x224_nv12.bin`
- [ ] `model_sha256.txt`
- [ ] 板端性能日志
- [ ] Foxglove 截图或录屏

## 最终结论

- [ ] 通过
- [ ] 未通过

- 未通过项：
