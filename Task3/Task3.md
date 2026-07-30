# Task3：X5 OpenExplorer PTQ 量化与 ROS 2 板端部署

## 1. 任务定位

Task3 接收 Task2 导出的浮点 ONNX 和冻结数据集，在 Windows WSL2 的 Ubuntu
环境中使用 X5 OpenExplorer v1.2.8 完成模型检查、校准数据准备、PTQ 量化、
量化精度评估和 BIN 编译；随后把 BIN 与 `racing_track_detection` ROS 2
节点部署到 RDK X5，并通过 Foxglove ImageAnnotations 查看检测结果。

整体工程流：

```text
Task2 float ONNX + frozen manifest
  → 固结模型与数据基线
  → hb_mapper checker
  → 从 train/val 选择代表性校准样本
  → RGB CHW float32 calibration binaries
  → hb_mapper makertbin
  → quantized ONNX / BIN
  → 浮点与量化任务指标对比
  → hb_verifier / hb_model_info / hb_perf
  → RDK X5 板端加载和实测
  → ROS 2 检测话题
  → Foxglove ImageAnnotations
```

最终交付必须同时证明：

1. 转换配置与 Task2 预处理一致。
2. 量化前后任务精度变化可接受。
3. 量化 ONNX 与 BIN 一致。
4. BIN 能在 X5 BPU 上加载并稳定运行。
5. ROS 2 坐标与 Task2/控制器契约一致。
6. Foxglove 中的点位、置信度和时间同步正确。

## 2. 模型、图像与部署契约

以下约定是 Task2、OpenExplorer、板端检测节点和控制器之间的共同接口。任何一项
发生变化，都必须同步修改 PTQ YAML、量化评估脚本、C++ 后处理、
ImageAnnotations 发布和验收报告。

| 契约项 | 当前约定 | Task3 含义 |
| --- | --- | --- |
| 浮点模型 | ONNX opset 11，固定 batch | 禁止用动态输入或未经 Task2 一致性验证的临时模型替换 |
| ONNX 输入 | 名称 `input`，`[1,3,224,224]`，RGB、NCHW、float32 | 浮点 ONNX Runtime 侧使用 Task2 的 ImageNet 预处理 |
| ONNX 输出 | 名称 `output`，`[1,3]`，顺序 `[x_norm,y_norm,track_logit]` | 前两项反归一化坐标，第三项执行 sigmoid |
| ROI | 640×480 标准图像底部 `frame[256:480,0:640]` | 板端对任意实际分辨率裁剪相同比例的底部区域 |
| ROI 坐标 | 左上原点，x 向右，y 向下 | `x=(x_norm+1)×320`，`y_roi=(y_norm+1)×112` |
| 控制坐标 | 640×480 | `y_full=y_roi+256`；禁止再次上下翻转 |
| 校准输入 | RGB、CHW、`[3,224,224]`、float32、`[0,255]` | 不先除 255，不先做 mean/std；由 OpenExplorer 配置注入 |
| BPU 运行输入 | NV12、224×224、`uint8` | `input_type_rt=nv12`，板端直接消费 Task1 NV12 shared memory |
| X5 架构 | `bayes-e` | 不能使用 X3/J3 的 `bernoulli2` 或 J5 的 `bayes` |
| ROS 2 输入 | `/image_hbmem` | 与 Task1 相机附件一致 |
| ROS 2 输出 | `/racing_track_center_detection` | `ai_msgs/msg/PerceptionTargets`，输出 640×480 坐标和 confidence |
| 图像标注 | `/racing_track_annotations` | `foxglove_msgs/msg/ImageAnnotations`，与 `/image_show` 按时间戳同步 |

OpenExplorer 配置中的预处理等价于：

```text
normalized[channel] =
  (RGB_0_to_255[channel] - mean_value[channel]) * scale_value[channel]
```

其中：

```text
mean_value  = [123.675, 116.28, 103.53]
scale_value = [
  1 / (255 × 0.229),
  1 / (255 × 0.224),
  1 / (255 × 0.225)
]
```

这与 Task2 的 `ToTensor()` 加 ImageNet mean/std 等价。不要把 mean 写成
`[0.485,0.456,0.406]` 同时又提供 `[0,255]` 校准数据，也不要把已归一化数据
交给 OpenExplorer 再归一化一次。

## 3. 学习目标

完成 Task3 后，你应能：

1. 说明 Windows、WSL、Docker 容器和 RDK X5 的职责边界。
2. 在启用 systemd 的 WSL2 Ubuntu 中安装和管理原生 Docker Engine。
3. 使用离线镜像进入固定版本的 OpenExplorer v1.2.8 环境。
4. 使用 `hb_mapper checker` 检查 ONNX 对 X5 的可转换性。
5. 解释 PTQ 校准集为什么需要代表性、为什么不使用 test。
6. 生成符合 RGB/CHW/float32/[0,255] 契约的校准二进制。
7. 说明 `input_type_train=rgb` 与 `input_type_rt=nv12` 的区别。
8. 使用 `hb_mapper makertbin` 生成量化 ONNX 和 X5 BIN。
9. 对比浮点和量化模型的任务指标、逐样本坐标漂移与置信度变化。
10. 使用 `hb_model_info`、`hb_perf`、`hb_verifier` 和板端工具验证产物。
11. 部署 `racing_track_detection` ROS 2 节点。
12. 在 Foxglove 中确认摄像头、预测点、置信度和时间同步。

## 4. 任务边界

### 4.1 Task3 输入

来自 Task2 的最低输入：

```text
Task2/
├── data/
│   └── dataset_manifest.csv
├── output/
│   ├── line_follower_resnet18.onnx
│   ├── onnx_parity_report.json
│   └── evaluation/
│       └── metrics.json
└── attachments/track_detection_resnet18/configs/
    ├── model_contract.yaml
    └── train_resnet18_xyv.yaml
```

正式量化前必须冻结：

- ONNX SHA256。
- dataset manifest SHA256。
- train config。
- test split。
- 浮点 test 指标。
- 置信度阈值。

Task3 不负责通过改变 test、删除困难样本或更换标签来修饰量化结果。若 Task2
数据或模型不满足最终门槛，应回到 Task2 补数据、重训并重新冻结交付。

### 4.2 Task3 输出

建议提交目录：

```text
Task3_submission_<组别>/
├── acceptance_report.md
├── environment_report.json
├── input_freeze/
│   ├── onnx_sha256.txt
│   ├── dataset_manifest_sha256.txt
│   ├── train_config.yaml
│   └── float_test_metrics.json
├── calibration/
│   ├── calibration_manifest.csv
│   └── calibration_report.json
├── conversion/
│   ├── resnet18_xyv.yaml
│   ├── hb_mapper_checker.log
│   ├── hb_mapper_makertbin.log
│   ├── hb_model_info.log
│   ├── hb_perf.log
│   ├── hb_verifier.log
│   ├── race_track_detection_224x224_nv12_quantized_model.onnx
│   ├── race_track_detection_224x224_nv12.bin
│   └── model_sha256.txt
├── quantized_evaluation/
│   ├── metrics.json
│   └── predictions.csv
├── board/
│   ├── model_info.txt
│   ├── hrt_model_exec_perf.txt
│   ├── ros2_topic_hz.txt
│   └── runtime_monitor.txt
└── foxglove/
    ├── straight.png
    ├── left_turn.png
    ├── right_turn.png
    └── no_track.png
```

校准 `.rgb` 文件、中间 ONNX、大体积 BIN 和运行日志默认放在
`Task3/output/`，不提交到 Git；最终提交包按验收要求单独整理。

## 5. 配套文件

| 文件 | 用途 |
| --- | --- |
| `attachments/openexplorer/` | WSL Docker、校准、PTQ、评估和 BIN 验证工具 |
| `attachments/openexplorer/config/resnet18_xyv.yaml` | X5 PTQ 主配置 |
| `attachments/openexplorer/README.md` | 工具顺序与数据契约 |
| `attachments/racing_track_detection/` | RDK X5 ROS 2 部署包 |
| `attachments/racing_track_detection/README.md` | 板端构建、话题与启动说明 |
| `templates/acceptance_report.md` | Task3 验收报告模板 |
| `output/` | 本地生成目录；除 `.gitkeep` 外不提交 |

本文默认从 Windows Terminal 或 PowerShell 进入 WSL2 Ubuntu：

```powershell
wsl --list --verbose
wsl -d Ubuntu-22.04
```

进入 WSL 后从仓库根目录执行：

```bash
cd ~/DRobotRookieTask
```

容器内仓库固定挂载为：

```text
/workspace
```

## 6. 任务实施

以下阶段严格按 A 到 F 完成。每个阶段的退出条件满足后，才进入下一阶段。

### 阶段 A：安装 Docker 并准备 OpenExplorer

OpenExplorer X5 转换工具以 Linux x86-64 Docker 镜像交付。
本任务使用 WSL2 内的原生 Docker Engine；如果采用此方案，不同时启用 Docker
Desktop 对同一 WSL 发行版的集成，避免命令连接到不同 daemon、镜像列表不一致
或端口冲突。

#### A1. 从 Windows 进入 WSL

在 Windows Terminal 或 PowerShell 中：

```powershell
wsl -d Ubuntu-22.04
```

进入 Ubuntu 后：

```bash
uname -a
cat /etc/os-release
ps -p 1 -o comm=
```

通过条件：

- 系统为 Linux x86-64。
- Ubuntu 为 22.04。
- PID 1 为 `systemd`。
- 仓库位于 WSL Linux home，不在 `/mnt/c`。

若 systemd 未启用，在 WSL 的 `/etc/wsl.conf` 中配置：

```ini
[boot]
systemd=true
```

然后从 Windows 执行 `wsl --shutdown`，重新启动发行版。

#### A2. 安装 WSL 原生 Docker Engine

附件脚本使用 Docker 官方 Ubuntu apt 仓库；若发现可能冲突的软件包会停止，
不会自动删除已有包。

```bash
cd ~/DRobotRookieTask
bash Task3/attachments/openexplorer/scripts/install_docker_wsl.sh
```

脚本结束后输入 `exit` 返回 Windows，再重新进入 WSL，使 docker group 生效：

```powershell
wsl -d Ubuntu-22.04
```

进入 WSL 后检查：

```bash
docker version
docker info
systemctl is-active docker
docker run --rm hello-world
```

通过条件：

- client 和 server 都能返回版本。
- Docker daemon 为 `active`。
- 普通用户执行 Docker 不需要每次 `sudo`。
- `hello-world` 成功。

#### A3. 准备 X5 v1.2.8 发布物

本任务固定以下四个文件，并统一存放在仓库根目录 `Downloads/`。先进入仓库：

```bash
cd ~/DRobotRookieTask
mkdir -p Downloads

# SDK
wget -c -P Downloads https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz

# 文档
wget -c -P Downloads https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-cn.zip

# CPU 离线镜像
wget -c -P Downloads https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz

# Release note
wget -c -P Downloads https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/release_note_CN.txt
```

附件脚本默认复用
`~/DRobotRookieTask/Downloads/` 中字节数正确的文件；文件缺失或下载不完整时才断点续传：

```bash
bash Task3/attachments/openexplorer/scripts/prepare_openexplorer.sh
```

注意：SDK 文件名虽然以 `.tar.gz` 结尾，该版本发布物可能是未 gzip 压缩的 POSIX
tar。附件使用 `tar -xf` 自动识别，不使用固定的 `tar -xzf`。

脚本完成：

- SDK 解包到 `~/open_explorer`。
- 加载离线 CPU Docker 镜像。
- 输出镜像 tag 和 image ID。
- 保留本地文档与 release note。

预期镜像：

```text
openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310
```

#### A4. 进入固定容器

```bash
bash Task3/attachments/openexplorer/scripts/enter_openexplorer.sh
```

进入后：

```bash
pwd
python3 --version
hb_mapper --version
hb_mapper --help
hb_model_info --help
```

预期：

- 当前目录为 `/workspace`。
- `/workspace/Task2` 和 `/workspace/Task3` 可见。
- `/open_explorer` 为解包后的 SDK。
- `hb_mapper`、`hb_model_info` 可用。
- release note 对应 OpenExplorer 1.2.8。

官方 CPU 镜像默认以 root 写入挂载目录。全部转换完成并退出容器后，如果 WSL
普通用户无法修改输出，可在仓库根目录修正 Task3 生成物的属主：

```bash
sudo chown -R "$(id -u):$(id -g)" Task3/output
```

退出条件：保存 Docker image ID、工具版本、Python 版本和命令输出到验收报告。

### 阶段 B：冻结并检查浮点模型

模型检查分两层：

```text
结构契约检查
  ├─ 输入输出名称、形状、类型
  ├─ opset
  ├─ onnx.checker
  └─ SHA256

X5 可转换性检查
  ├─ 算子支持
  ├─ shape 推断
  ├─ BPU/CPU 划分
  └─ 转换阻断错误
```

#### B1. Task2 交付门禁

正式 PTQ 前检查：

- `Task2/output/line_follower_resnet18.onnx` 存在且非空。
- `onnx_parity_report.json` 通过。
- float test metrics 已生成。
- train/val/test 都有已审核正、负样本。
- 同一视频不跨 split。
- 当前 ONNX SHA 与交接记录一致。

任一门禁不满足时停止 Task3，返回 Task2 补充数据、重新训练和导出。

#### B2. 执行检查

在 OpenExplorer 容器内：

```bash
cd /workspace
bash Task3/attachments/openexplorer/scripts/01_check_model.sh
```

生成：

```text
Task3/output/environment_report.json
Task3/output/logs/hb_mapper_checker.log
```

通过条件：

- 平台为 Linux x86-64。
- ONNX input 为 `input [1,3,224,224]`。
- ONNX output 为 `output [1,3]`。
- opset 为 11。
- `march=bayes-e`。
- `hb_mapper checker` 无阻断错误。
- 不存在意外的大段 CPU fallback。

`checker` 通过只表示模型可转换，不代表量化精度通过。若发生 CPU fallback，必须
在报告中记录节点、原因和预期性能影响。

### 阶段 C：准备校准数据集

PTQ 使用代表性数据估计激活值范围。校准数据不要求标签，但必须覆盖真实部署
分布；冻结 test 只用于最后评估，不能参与选择量化参数。

#### C1. 选择原则

默认选择 64 张，来源仅为 train/val：

- 覆盖不同 `video_id`。
- 覆盖顺时针、逆时针、直道和弯道。
- 覆盖正常、偏暗、局部强光。
- 覆盖居中、左偏、右偏。
- 正式模型应包含有赛道和无赛道画面。
- 排除严重模糊、错误曝光、错误 ROI、损坏图片和 rejected 样本。
- 不使用 test。

脚本按 `video_id + direction + lighting + annotation_status` 分层轮询抽取，使用
固定 seed，使选择可复现。

#### C2. 生成二进制

容器内执行：

```bash
bash Task3/attachments/openexplorer/scripts/02_prepare_calibration.sh
```

每个输出文件必须是：

```text
RGB
CHW
[3,224,224]
float32
0..255
602112 bytes
```

输出：

```text
Task3/output/
├── calibration_data_rgb_f32/
│   ├── 0000_<sample_id>.rgb
│   └── ...
├── calibration_manifest.csv
└── calibration_report.json
```

人工抽查：

```bash
find Task3/output/calibration_data_rgb_f32 \
  -maxdepth 1 -type f -name '*.rgb' | wc -l

find Task3/output/calibration_data_rgb_f32 \
  -maxdepth 1 -type f -name '*.rgb' \
  -printf '%s\n' | sort -u
```

预期数量为 64，唯一文件大小为 602112。

#### C3. 校准集门禁

正式验收必须确认：

- 没有 test 样本。
- manifest 中的每一行都能定位源图片。
- 输出文件 SHA256 已记录。
- 分层分布不是被单个视频垄断。
- 正负样本和关键场景有覆盖。
- 未执行二次归一化。

train/val 中必须包含代表性的正样本和负样本；脚本会把缺少任一类别视为错误，
不满足时返回 Task2 补充数据。

### 阶段 D：PTQ、量化评估与 BIN 验证

#### D1. 检查 PTQ YAML

主配置：

```text
Task3/attachments/openexplorer/config/resnet18_xyv.yaml
```

关键字段：

```yaml
model_parameters:
  march: "bayes-e"

input_parameters:
  input_type_rt: "nv12"
  input_type_train: "rgb"
  input_layout_train: "NCHW"
  norm_type: "data_mean_and_scale"

calibration_parameters:
  cal_data_type: "float32"
  preprocess_on: false
  calibration_type: "default"

compiler_parameters:
  compile_mode: "latency"
  optimize_level: "O3"
```

`preprocess_on: false` 表示校准文件已经由附件处理到 224×224 RGB CHW
float32。运行时 NV12 到训练 RGB 的颜色转换由转换后的模型前端与 X5 硬件路径
完成。

#### D2. 执行 PTQ

```bash
bash Task3/attachments/openexplorer/scripts/03_build_model.sh
```

主要输出：

```text
Task3/output/model_output/
├── race_track_detection_224x224_nv12_original_float_model.onnx
├── race_track_detection_224x224_nv12_optimized_float_model.onnx
├── race_track_detection_224x224_nv12_calibrated_model.onnx
├── race_track_detection_224x224_nv12_quantized_model.onnx
├── race_track_detection_224x224_nv12.bin
└── *_subgraph_*.html/json
```

日志：

```text
Task3/output/logs/hb_mapper_makertbin.log
```

通过条件：

- 命令退出码为 0。
- 量化 ONNX 和 BIN 均非空。
- 没有 NaN、Inf 或量化失败。
- 输出 shape 仍为 `[1,3]` float。
- BPU 子图和静态性能报告已生成。

不要对模型最终三个输出启用整数输出优化；ROS 2 后处理按三个 float 读取。

#### D3. 检查 BIN 与静态性能

```bash
bash Task3/attachments/openexplorer/scripts/04_inspect_bin.sh
```

检查 `hb_model_info.log`：

- march 为 `bayes-e`。
- 输入为 224×224 NV12。
- 输出元素数为 3。
- 模型版本和 SDK 版本已记录。

`hb_perf` 是开发机静态估算，不等于板端真实 latency，也不等于 ROS 2 端到端
FPS，三者都需要分别记录。

#### D4. 量化精度评估

正式模型：

```bash
bash Task3/attachments/openexplorer/scripts/05_evaluate_quantized.sh
```

脚本对同一冻结 test 图像执行：

```text
PIL RGB resize + ImageNet normalize
  → float ONNX Runtime

OpenCV BGR resize → NV12 → YUV444 intermediate
  → HB_ONNXRuntime quantized ONNX
```

输出：

```text
Task3/output/quantized_evaluation/
├── metrics.json
└── predictions.csv
```

正式通过门槛：

| 指标 | 门槛 |
| --- | ---: |
| 评估样本数 | ≥ 50 |
| 平均量化坐标漂移 | ≤ 2 px |
| P95 量化坐标漂移 | ≤ 5 px |
| 量化任务距离 MAE | 不超过浮点 `+ max(2 px, 10%)` |
| 置信度 F1 下降 | 有正负样本时 ≤ 0.02 |
| NaN/Inf | 0 |

#### D5. 量化 ONNX 与 BIN 一致性

```bash
bash Task3/attachments/openexplorer/scripts/06_verify_bin.sh
```

该步骤回答“量化 ONNX 和最终 BIN 是否保持数值一致”，不能替代 D4 的任务精度
评估。

#### D6. 量化不通过时

按以下顺序定位，不先修改 test：

1. 检查 RGB/BGR、NHWC/NCHW、值域和 mean/std。
2. 检查校准 `.rgb` 是否正好 602112 bytes。
3. 检查校准场景是否代表部署分布。
4. 增加 train/val 的代表性校准样本，不使用 test。
5. 对比逐样本 `predictions.csv`，定位量化敏感场景。
6. 先使用官方推荐的 `calibration_type=default`；仍不满足时再按 v1.2.8
   精度调优文档尝试 mix、敏感节点或 int16。
7. 每次更改生成新的配置 SHA、模型 SHA 和报告，不覆盖已验收基线。

### 阶段 E：部署 racing_track_detection ROS 2 节点

#### E1. 在 WSL 暂存 BIN

先把生成模型复制到包内；该文件被 `.gitignore` 排除，但会随包复制到板端：

```bash
cd ~/DRobotRookieTask
bash Task3/attachments/racing_track_detection/scripts/stage_model.sh
```

确认源和目标 SHA256 相同。

#### E2. 复制到 RDK X5

示例从 WSL 执行；按实际板端 IP 修改：

```bash
BOARD=root@<RDK_X5_IP>

scp -r Task3/attachments/racing_track_detection \
  "$BOARD":/root/D_Robot/dev_ws/src/
```

如果板端已有同名目录，先只读比较差异和模型 SHA，再决定如何更新；不要直接删除
整个工作空间。

#### E3. 板端构建

登录板端：

```bash
ssh root@<RDK_X5_IP>
source /opt/tros/humble/setup.bash
ros2 interface show foxglove_msgs/msg/ImageAnnotations
```

如果找不到消息包：

```bash
sudo apt update
sudo apt install ros-humble-foxglove-msgs
```

然后构建：

```bash
cd /root/D_Robot/dev_ws
source /opt/tros/humble/setup.bash
colcon build --packages-select racing_track_detection --symlink-install
source install/setup.bash
```

检查：

```bash
ros2 pkg prefix racing_track_detection
find install/racing_track_detection \
  -name 'race_track_detection_224x224_nv12.bin' -ls
```

#### E4. 板端 BIN 性能

在开始 ROS 2 之前先单独测模型：

```bash
MODEL=/root/D_Robot/dev_ws/install/racing_track_detection/share/racing_track_detection/config/race_track_detection_224x224_nv12.bin

hrt_model_exec model_info --model_file="$MODEL"
hrt_model_exec perf \
  --model_file "$MODEL" \
  --thread_num 1 \
  --frame_count 1000
```

记录：

- 单线程 latency。
- FPS。
- BPU core。
- 内存。
- 是否有 runtime/模型版本不兼容。

#### E5. 静止状态启动摄像头和检测

第一轮测试必须让车轮离地或底盘保持停止，不启动控制闭环。

终端 1：

```bash
cd /root/D_Robot/dev_ws
source /opt/tros/humble/setup.bash
source install/setup.bash
ros2 launch origincar_bringup camera.launch.py
```

终端 2：

```bash
cd /root/D_Robot/dev_ws
source /opt/tros/humble/setup.bash
source install/setup.bash
ros2 launch racing_track_detection racing_track_detection.launch.py
```

终端 3：

```bash
ros2 topic hz /image_hbmem
ros2 topic hz /racing_track_center_detection
ros2 topic hz /racing_track_annotations
ros2 topic echo /racing_track_center_detection --once
```

手持车辆或赛道分别观察：

- 直道中心。
- 左弯。
- 右弯。
- 赛道离开视野。
- 部分遮挡。
- 暗光和强光。

坐标检查：

```text
0 ≤ x ≤ 640
256 ≤ y ≤ 480
0 ≤ confidence ≤ 1
header.stamp 与图像时间一致
```

`/sign4return` 行为：

```bash
ros2 topic pub --once /sign4return std_msgs/msg/Int32 '{data: 5}'
ros2 topic pub --once /sign4return std_msgs/msg/Int32 '{data: 6}'
```

5 暂停巡线推理，6 恢复。

#### E6. 低速闭环

只有以下项目通过后才能启动底盘控制：

- 摄像头和检测至少稳定运行 5 分钟。
- 坐标方向正确。
- 置信度低时控制器能停车或进入安全策略。
- sign 5 能停止巡线行为。
- 紧急停止方式已验证。

初次闭环速度不超过 `0.3 m/s`。先直线、停车、缓弯，再完整赛道。Task3
验收关注部署正确性和安全，不要求追求极限速度。

### 阶段 F：使用 Foxglove ImageAnnotations

`racing_track_detection` 在发布检测结果的同时直接发布：

```text
/racing_track_annotations
  foxglove_msgs/msg/ImageAnnotations
```

每条消息包含：

- 预测点圆圈。
- confidence 文本。
- 与输入 NV12 图像一致的时间戳。
- 按原始相机分辨率缩放后的像素坐标。

不创建第二条 JPEG 图像流，也不需要单独启动叠加节点。

#### F1. 检查 ImageAnnotations

检测节点已经在 E5 启动。另开板端终端：

```bash
ros2 topic info /racing_track_annotations -v
ros2 topic hz /racing_track_annotations
ros2 topic echo /racing_track_annotations --once
```

确认 `circles` 和 `texts` 均非空，circle 与 text 的 timestamp 相同。

#### F2. 启动 Foxglove Bridge

另一个板端终端：

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

Windows Foxglove Desktop 连接：

```text
ws://<RDK_X5_IP>:8765
```

在 Image 面板中：

1. 主图像选择 `/image_show`。
2. 在 **Image annotations** 中启用 `/racing_track_annotations`。
3. 启用 **Sync timestamps**。

再添加 Raw Messages 面板查看 `/racing_track_center_detection`。Foxglove 会根据
相同 timestamp，把圆圈和 confidence 文本直接绘制在 `/image_show` 上。

#### F3. 视觉验收

至少保存四张截图或一段录屏：

1. 直道中心。
2. 左弯或右弯。
3. 车辆横向偏离。
4. 无赛道或遮挡。

逐项确认：

- 点位没有左右镜像。
- 点位没有上下翻转。
- 预测点与原始赛道目标位置一致。
- confidence 文本与 Raw Messages 中的数值一致。
- 启用同步后，图像与标注没有明显错帧。
- ImageAnnotations 不拖慢 `/image_hbmem` 的 28～30 FPS。

如果 ImageAnnotations 正确而控制错误，检查控制器的坐标和阈值；如果检测消息
与标注都错误，检查模型、ROI 和后处理；如果检测消息正确但标注位置错误，检查
原始图像尺寸缩放和时间戳。

## 7. 验收与提交

### 7.1 硬门槛

| 类别 | 验收项 | 通过条件 |
| --- | --- | --- |
| 环境 | WSL | Ubuntu 22.04、Linux x86-64、systemd |
| 环境 | Docker | WSL 原生 daemon 可复现运行 |
| 环境 | OpenExplorer | v1.2.8 CPU 镜像与 image ID 有记录 |
| 输入 | ONNX | 契约、opset、SHA 和 Task2 parity 全部通过 |
| 输入 | 数据 | train/val/test 都有已审核正负样本 |
| 校准 | 来源 | 只使用 train/val |
| 校准 | 格式 | 64 个 RGB CHW float32 `[0,255]` 文件 |
| 检查 | checker | 无阻断错误 |
| PTQ | 产物 | 量化 ONNX、BIN、日志和静态报告齐全 |
| 精度 | 漂移 | mean ≤ 2 px，P95 ≤ 5 px |
| 精度 | 任务指标 | MAE/F1 退化在门槛内 |
| 一致性 | verifier | 量化 ONNX/BIN 通过 |
| 板端 | runtime | BIN 可加载，性能日志完整 |
| ROS 2 | 坐标 | 640×480、左上原点、y 向下、ROI +256 |
| ROS 2 | 稳定性 | 运行 5 分钟无崩溃或持续掉帧 |
| Foxglove | ImageAnnotations | 四类场景截图/录屏及时间同步正确 |
| 安全 | 闭环 | 静止验证后低速测试，停车措施有效 |

任一 split 缺少正样本或负样本时，`final_acceptance` 不通过。

### 7.2 性能记录

最低记录三类性能，不能互相替代：

| 层次 | 工具 | 回答的问题 |
| --- | --- | --- |
| 编译静态估算 | `hb_perf` | BPU 子图理论性能和带宽 |
| 板端模型 | `hrt_model_exec perf` | BIN 在 X5 runtime 的实际模型性能 |
| ROS 2 端到端 | `ros2 topic hz` | 相机、预处理、调度、后处理和发布后的实际频率 |

目标是 ROS 2 检测话题持续接近相机推理流，且 ImageAnnotations 发布不造成明显
下降。如果模型端很快但 ROS 2 很慢，应检查日志级别、内存复制、topic QoS 和
CPU 使用率。

### 7.3 提交

复制 `templates/acceptance_report.md` 并完成全部字段：

```bash
cp Task3/templates/acceptance_report.md \
  Task3/output/acceptance_report.md
```

报告中的每个“通过”必须附路径、日志、SHA、数值或截图；仅写“已完成”不算证据。

## 8. 实施里程碑

| 里程碑 | 内容 | 退出条件 |
| --- | --- | --- |
| M1 | WSL 原生 Docker | daemon、普通用户、hello-world 通过 |
| M2 | OpenExplorer v1.2.8 | 固定镜像和工具版本可复现 |
| M3 | ONNX 冻结与 checker | 契约、SHA、算子检查通过 |
| M4 | 校准集 | 64 个样本、格式和追溯报告通过 |
| M5 | PTQ | 量化 ONNX、BIN、转换日志齐全 |
| M6 | 量化精度 | 漂移与任务指标通过；正式模式有负样本 |
| M7 | BIN 验证 | info、perf、verifier 通过 |
| M8 | ROS 2 板端 | 坐标、频率、稳定性通过 |
| M9 | Foxglove | 四类场景 ImageAnnotations 证据齐全 |
| M10 | 低速闭环 | 安全策略和低速路线验证通过 |

## 9. 当前附件脚本

### 9.1 OpenExplorer

| 顺序 | 文件 | 作用 |
| ---: | --- | --- |
| 1 | `scripts/install_docker_wsl.sh` | 安装 WSL 原生 Docker CE |
| 2 | `scripts/prepare_openexplorer.sh` | 发布物下载、SDK 解包和离线镜像加载 |
| 3 | `scripts/enter_openexplorer.sh` | 固定挂载仓库和 SDK 并进入容器 |
| 4 | `check_environment.py` | 环境、ONNX 和 YAML 契约检查 |
| 5 | `scripts/01_check_model.sh` | 运行环境检查与 `hb_mapper checker` |
| 6 | `prepare_calibration.py` | 分层选择并生成校准二进制 |
| 7 | `scripts/02_prepare_calibration.sh` | 生成默认 64 样本 |
| 8 | `scripts/03_build_model.sh` | 执行 PTQ 与 BIN 编译 |
| 9 | `scripts/04_inspect_bin.sh` | 模型信息、静态性能和 SHA |
| 10 | `evaluate_quantized.py` | 浮点/量化精度与漂移比较 |
| 11 | `scripts/05_evaluate_quantized.sh` | 正式 test 验收 |
| 12 | `scripts/06_verify_bin.sh` | 量化 ONNX/BIN 一致性 |

### 9.2 ROS 2

| 文件 | 作用 |
| --- | --- |
| `src/racing_track_detection.cpp` | NV12 ROI、BPU 推理、坐标解码与 ImageAnnotations |
| `scripts/stage_model.sh` | 将生成的 BIN 暂存到包内 |
| `launch/racing_track_detection.launch.py` | 启动检测和 ImageAnnotations 发布 |
| `CMakeLists.txt` / `package.xml` | ROS 2 构建和依赖 |

## 10. 相关资料

- [D-Robotics：X5 算法工具链环境安装](https://developer.d-robotics.cc/rdk_x_doc/Advanced_development/toolchain_development/intermediate/environment_config)
- [D-Robotics：X5 OpenExplorer 文档入口](https://developer.d-robotics.cc/api/v1/fileData/x5_doc-v126cn/index.html)
- [Docker：在 Ubuntu 安装 Docker Engine](https://docs.docker.com/engine/install/ubuntu/)
- [Docker：Linux 安装后操作](https://docs.docker.com/engine/install/linux-postinstall/)
- [Foxglove：ImageAnnotations 消息](https://docs.foxglove.dev/docs/sdk/schemas/image-annotations)
- [Foxglove：Image 面板与时间戳同步](https://docs.foxglove.dev/docs/visualization/panels/image)
