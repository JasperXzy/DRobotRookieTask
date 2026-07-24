# Task2：巡线数据集制作与 ResNet18 浮点模型训练

## 1. 任务定位

Task2 接收 Task1 采集的 MP4 视频和元数据，在 Windows WSL2 的 Ubuntu
22.04 环境中，离线完成视频校验、抽帧、巡线目标点标注、数据集划分、
ResNet18 训练、测试视频评估和 ONNX 导出。


整体数据流：

```text
Task1 MP4/JSON
  → 视频校验
  → 按时间抽帧
  → 640×480 统一尺寸
  → 底部 640×224 ROI
  → 数据集标注
  → 数据集验证
  → 按视频划分 train/val/test
  → ResNet18 训练
  → 独立图片和视频评估
  → ONNX 导出与一致性验证
```

## 2. 数据与模型契约

以下约定已经落实到 Task2 的配置和脚本中，是数据制作、浮点训练、ONNX
导出及 Task3 部署之间的共同接口。附件中的
`Task2/attachments/track_detection_resnet18/configs/model_contract.yaml`
是机器可读的契约文件；修改下表任一项时，必须同步更新该文件、相关脚本和
验收报告。

| 契约项 | 当前约定 | 说明 |
| --- | --- | --- |
| 图像与 ROI | 原始帧先缩放到 `640×480`，再裁剪 `frame[256:480, 0:640]`，得到 `640×224` ROI | 保证数据制作、训练和部署使用相同视野 |
| 坐标系 | ROI 左上为原点，x 向右、y 向下；坐标归一化到 `[-1,1]` | 避免训练、可视化和部署之间发生轴翻转或重复归一化 |
| 标注空间 | 在 `640×224` ROI 上保存原始像素坐标，不在 `224×224` 模型输入上标注 | 保留人工标注精度，归一化由数据集加载器统一完成 |
| 模型输入 | `[1,3,224,224]`、RGB、NCHW、float32，使用 ImageNet mean/std | PyTorch、ONNX 和 Task3 必须采用完全一致的预处理 |
| 模型输出 | `[x_norm, y_norm, track_logit]`，形状为 `[1,3]`、float32 | 坐标表示目标点，`sigmoid(track_logit)` 表示赛道置信度 |

## 3. 学习目标

完成 Task2 后，你应能：

1. 安装和检查 WSL2 Ubuntu 22.04。
2. 使用 Conda 创建隔离的 Python 3.10/PyTorch 训练环境。
3. 说明为什么 Task1 的 MP4 和同名 JSON 足以支持本任务的离线流程。
4. 说明 ROI、RGB/BGR、resize、NCHW 和 ImageNet 归一化的含义。
5. 从视频中按固定时间间隔抽帧，并保留原视频、帧号和时间戳关系。
6. 使用交互式点标注脚本标记巡线目标点和无有效轨迹样本。
7. 说明为什么相邻帧不能随机分散到 train、val 和 test。
8. 训练两输出教学基线和三输出最终模型。
9. 使用像素误差、P90/P95 和置信度 F1 评价模型。
10. 导出固定输入的 ONNX，并验证 PyTorch 与 ONNX Runtime 输出一致。
11. 整理可交给 Task3 的模型、配置、哈希和报告。

## 4. 任务边界

### 4.1 Task2 输入

来自 Task1：

```text
Task1/output/
├── task1_line_001.mp4
├── task1_line_001.json
├── task1_line_002.mp4
├── task1_line_002.json
└── ...
```

以及填写完成的 `recording_manifest.csv` 副本。

每段视频至少需要提供：

- 小车编号。
- 任务、路线和圈次。
- 顺时针或逆时针。
- 光照条件。
- 相机安装状态。
- 障碍物或特殊场景说明。
- 视频分辨率、FPS 和时长。

### 4.2 Task2 输出

```text
Task2_submission_<组别>/
├── acceptance_report.md
├── environment_report.json
├── dataset_card.md
├── video_manifest.csv
├── dataset_manifest.csv
├── split_manifest.csv
├── train_config.yaml
├── model_contract.yaml
├── environment.yml
├── environment-lock.yml
├── requirements-lock.txt
├── best_model.pth
├── line_follower_resnet18.onnx
├── evaluation_report.md
├── onnx_parity_report.json
└── predictions/
    ├── test_images/
    └── test_video.mp4
```


## 5. 配套文件

| 文件 | 用途 |
| --- | --- |
| `attachments/track_detection_resnet18/` | Task2 完整数据处理、训练、评估和 ONNX 工具包 |
| `attachments/track_detection_resnet18/README.md` | 附件文件说明和运行约定 |
| `attachments/track_detection_resnet18/configs/` | 数据集、训练和模型契约配置 |
| `attachments/track_detection_resnet18/requirements.txt` | 不包含 PyTorch 的通用 Python 依赖 |
| `attachments/labeling_guide.md` | 点标注脚本使用与标注规范 |
| `templates/acceptance_report.md` | Task2 验收报告模板 |
| `templates/video_manifest.csv` | Task1 视频导入清单模板 |
| `templates/dataset_manifest.csv` | 帧、标签、来源和 split 清单模板 |
| `templates/split_manifest.csv` | 视频级数据集划分模板 |
| `output/` | 本地生成结果目录；除 `.gitkeep` 外不提交 |

本文所有命令都从仓库根目录执行。进入 Conda 环境后建议先定义：

```bash
conda activate drobot-train
TRAIN_ROOT=~/DRobotRookieTask/Task2/attachments/track_detection_resnet18
```

## 6. 任务实施

以下阶段按 A 到 K 顺序完成，每个阶段先说明相关概念和这样设计的原因，再给出
操作步骤、命令与通过条件。

### 阶段 A：安装 WSL2 Ubuntu 22.04

WSL2 让 Windows 可以运行完整的 Linux 环境，模型训练会频繁读写大量小图片、
标签和日志，因此仓库、数据集和训练输出都放在 WSL2 的 Linux home 目录中；
通过 `/mnt/c` 跨 Windows 与 Linux 文件系统访问这些小文件，通常会带来明显的
I/O 性能损失。

#### A1. Windows 前置检查

推荐 Windows 11；Windows 10 需要满足 WSL2 支持要求，确认系统已启用硬件
虚拟化，以下命令在管理员 PowerShell 中执行。

先查看在线发行版的准确名称：

```powershell
wsl --list --online
```

安装列表中的 Ubuntu 22.04 发行版，发行版名称以命令实际输出为准，例如：

```powershell
wsl --install -d Ubuntu-22.04
```

按照提示重启 Windows，并在首次启动 Ubuntu 时创建普通 Linux 用户和密码。

检查状态：

```powershell
wsl --update
wsl --status
wsl --list --verbose
```

通过条件：

- Ubuntu 版本为 22.04。
- `wsl --list --verbose` 中 `VERSION` 为 `2`。
- 能进入 Ubuntu shell。
- 日常操作使用普通用户，不使用 root 作为默认用户。

#### A2. Ubuntu 基础工具

在 WSL2 Ubuntu 中执行：

```bash
sudo apt update
sudo apt install -y \
  ca-certificates \
  curl \
  git \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libice6
```


#### A3. 工作目录

直接在 WSL2 的 Linux home 目录中拉取培训仓库。以下命令必须在 WSL2
Ubuntu shell 中执行，并确认当前目录不是 `/mnt/c/...`：

```bash
cd ~
git clone https://github.com/JasperXzy/DRobotRookieTask.git
cd DRobotRookieTask
```

克隆后创建本地数据
目录：

```bash
mkdir -p \
  Task2/data/{raw,annotations,datasets} \
  Task2/data/frames/{full,roi} \
  Task2/runs
```

最终结构：

```text
~/DRobotRookieTask/
├── Task1/
├── Task2/
│   ├── data/
│   │   ├── raw/
│   │   ├── frames/
│   │   │   ├── full/
│   │   │   └── roi/
│   │   ├── annotations/
│   │   └── datasets/
│   ├── runs/
│   └── output/
└── ...
```

`Task2/data/`、`Task2/runs/` 和 `Task2/output/` 已通过 `.gitignore` 排除，
不会因为位于仓库目录内而被提交。仓库代码、数据和训练输出都位于 WSL2 的
Linux 文件系统，因此不会产生 `/mnt/c` 的大量小文件访问问题。

注意：`git clone` 只会取得代码、配置和模板。Task1 录制的视频已被 Git
忽略，不会随仓库下载，需要另外复制到 `Task2/data/raw/`。


### 阶段 B：配置 Conda 深度学习环境

这一阶段会接触五个容易混淆的概念：Conda 创建隔离环境并管理 Python 版本；
Python 是运行脚本的解释器；pip 在当前 Conda 环境内安装 Python 包；PyTorch
提供模型、损失函数和训练能力；CUDA 让支持的 NVIDIA GPU 加速 PyTorch。
它们职责不同，但共同组成训练环境。

#### B1. 安装 Miniconda

以下安装命令适用于常见的 WSL2 x86-64 环境，先执行 `uname -m`，结果应为
`x86_64`。

在 WSL2 Ubuntu 中执行：

```bash
cd ~
curl -O \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum Miniconda3-latest-Linux-x86_64.sh
```

将输出的 SHA256 与 Miniconda 官方下载目录公布的值比较，匹配后安装：

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程：

1. 阅读并接受安装许可。
2. 使用当前普通用户的默认安装目录。
3. 在询问是否执行 `conda init` 时选择 `yes`。
4. 安装完成后重新打开 Ubuntu shell，或执行 `source ~/.bashrc`。

检查并禁止每次打开 shell 自动进入 `base`：

```bash
conda --version
conda config --set auto_activate_base false
source ~/.bashrc
```

不得直接在 `base` 环境中安装 Task2 依赖或运行训练。

#### B2. 创建 Task2 环境

回到培训仓库根目录，根据仓库提供的环境文件创建 Conda 环境：

```bash
cd ~/DRobotRookieTask
conda env create -f Task2/environment.yml
conda activate drobot-train
```

检查当前解释器：

```bash
conda env list
echo "$CONDA_DEFAULT_ENV"
which python
python --version
```

通过条件：

- 当前环境名为 `drobot-train`。
- Python 版本为 3.10.x。
- `which python` 指向 Conda 的 `drobot-train` 环境，不是
  `/usr/bin/python`，也不是 `base`。

#### B3. 安装 PyTorch

PyTorch 安装命令取决于是否使用 NVIDIA GPU 以及当前官方提供的 CUDA
wheel。进入 PyTorch 官方安装选择器，选择：

```text
OS: Linux
Package: Pip
Language: Python
Compute Platform: CPU 或当前 WSL2 支持的 CUDA
```

复制选择器当前生成的命令执行，安装命令必须在 `drobot-train` 已激活时执行。CPU 环境允许完成任务，只是训练速度较慢。

#### B4. 安装通用依赖

```bash
conda activate drobot-train
python -m pip install -r "$TRAIN_ROOT/requirements.txt"
```

标注脚本使用 OpenCV 窗口，依赖已包含在附件的 `requirements.txt` 中，不再
安装额外的标注软件。

本任务采用以下依赖管理规则：

- Conda 只管理环境、Python 3.10 和 pip。
- PyTorch、torchvision 和项目依赖统一在该环境内使用 pip 安装。
- 不在 `base` 中安装项目依赖。

#### B5. GPU 检查

仅在控制电脑具有支持 CUDA 的 NVIDIA GPU 时执行 GPU 路径。首先在
PowerShell 和 WSL2 中确认 `nvidia-smi` 可用，再检查 PyTorch：

```bash
conda activate drobot-train
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

没有 NVIDIA GPU 时，`torch.cuda.is_available()` 为 `False` 是允许的。

#### B6. 环境验收

需要生成 `environment_report.json`，至少记录：

```json
{
  "ubuntu": "22.04",
  "conda": "",
  "conda_env": "drobot-train",
  "python": "3.10.x",
  "torch": "",
  "torchvision": "",
  "opencv": "",
  "onnx": "",
  "onnxruntime": "",
  "cuda_available": false,
  "gpu_name": null
}
```

通过条件：

- `conda env list` 中存在且已激活 `drobot-train`。
- Python 来自 `drobot-train`，不是系统 Python 或 `base`。
- OpenCV 能打开一个 Task1 MP4。
- PyTorch 能完成一次 ResNet18 前向推理。
- ONNX 和 ONNX Runtime 可以导入。
- `python "$TRAIN_ROOT/label_image.py" --help` 可以运行。
- 环境中不要求 ROS 2。

使用附件生成环境报告：

```bash
python "$TRAIN_ROOT/check_environment.py" \
  --video Task2/data/raw/task1_line_001.mp4
```

环境验证通过后保存实际依赖版本：

```bash
python -m pip check
conda env export --no-builds > Task2/output/environment-lock.yml
python -m pip freeze > Task2/output/requirements-lock.txt
```

`environment.yml` 用于声明环境名和 Python 版本，附件中的
`requirements.txt` 用于表达直接项目依赖，`environment-lock.yml` 和
`requirements-lock.txt` 用于记录本次已经验证过的完整环境。两份锁定文件
生成在 `Task2/output/`，整理提交包时复制到提交目录根部。

### 阶段 C：导入和校验 Task1 视频

MP4 保存实际图像帧；同名 JSON 保存分辨率、帧率、帧数等技术信息；
`recording_manifest.csv` 保存任务、路线、光照、行驶方向等场景信息，三者作用
不同，合在一起才能判断视频是否完整，并让后续样本追溯到采集条件。

#### C1. 原始数据只读原则

将 Task1 的 MP4 和同名 JSON 复制到 `Task2/data/raw/`，并将填写完成的
`recording_manifest.csv` 复制到 `Task2/data/recording_manifest.csv`。这些文件
都是只读输入；任何抽帧、筛选和重新编码都写入新的目录，不覆盖原始文件。

#### C2. 视频导入清单

将填写过的场景信息保存在视频清单模板副本中，然后使用附件校验
Task1 MP4/JSON，并为每段视频生成唯一 `video_id`：

```bash
python "$TRAIN_ROOT/import_task1_data.py" \
  --raw-dir Task2/data/raw \
  --recording-manifest Task2/data/recording_manifest.csv \
  --existing-manifest Task2/templates/video_manifest.csv \
  --output Task2/data/video_manifest.csv
```

当前 `import_task1_data.py` 会检查：

- MP4 和同名 JSON 是否同时存在。
- 视频能否解码首帧、中间帧和末帧。
- JSON 中的尺寸、帧数和 FPS 是否与视频基本一致。
- `recording_manifest.csv` 中是否有对应记录。
- 文件名是否唯一。
- 视频 SHA256 是否已记录。

任何后续样本都必须满足：

```text
sample_id → video_id → frame_index → timestamp_ms → 原始 MP4
```

### 阶段 D：视频抽帧和 ROI

30 FPS 视频每秒包含 30 张图像，相邻帧通常几乎相同，降到 2 FPS 可以减少重复
标注和数据相关性，同时保留车辆位置随时间的主要变化。ROI（Region of
Interest）是完整画面中真正用于巡线判断的区域。本任务先把原始帧统一缩放为
`640×480`，再按固定坐标裁剪 ROI，确保不同来源视频使用相同的图像几何和视野。

#### D1. 默认抽帧策略

默认使用：

```yaml
sample_fps: 2.0
```

30 FPS 视频不应全部导出用于人工标注，相邻帧高度相似，直接全量抽帧会增加
大量重复标注，并导致模型评估过于乐观。

#### D2. 统一图像几何

每帧按以下顺序处理：

```text
原始帧
  → 双线性缩放至 640×480
  → 裁剪 frame[256:480, 0:640]
  → 保存 640×224 ROI
```

附件中的 `extract_frames.py` 统一使用 Python 半开区间 `256:480`，
正好得到 224 行。

#### D3. 样本命名

```text
<video_id>__f<六位帧号>.jpg
```

例如：

```text
task1_line_001__f000090.jpg
```


#### D4. 质量标记

当前 `extract_frames.py` 会计算：

- Laplacian 方差 `blur_score`，用于标记 `blur`。
- 灰度均值 `brightness`，用于标记 `underexposed` 或 `overexposed`。
- 与前一张抽帧图片的感知哈希差异，用于标记 `near_duplicate`。

存在任一质量标记时，脚本将 `quality_status` 写为 `suspect`；否则写为
`accepted`，脚本不会因为质量标记自动删除图片。

当前输出：

```text
Task2/data/
├── frames/
│   ├── full/
│   └── roi/
└── frame_manifest.csv
```

执行当前附件脚本：

```bash
python "$TRAIN_ROOT/extract_frames.py" \
  --video-manifest Task2/data/video_manifest.csv \
  --config "$TRAIN_ROOT/configs/dataset.yaml" \
  --output-manifest Task2/data/frame_manifest.csv
```

### 阶段 E：标注数据集

正样本表示画面中存在可靠巡线目标点，会参与坐标损失和置信度损失；负样本
表示画面中没有可靠目标点，不提供坐标，但会以 `has_track=0` 参与置信度损失；
拒绝样本表示图片本身不适合使用，会从训练、验证和测试中排除。缺少标签的图片
只是尚未标注，不能自动当作负样本。

#### E1. 为什么先试标

在正式标注数千张图片前，先选取 100 张覆盖直线、左右弯、不同光照和车辆
偏差的图片。试标用于发现坐标、ROI 和标注规则中的问题。

#### E2. 启动点标注脚本

```bash
python "$TRAIN_ROOT/label_image.py" \
  --image-dir Task2/data/frames/roi \
  --label-dir Task2/data/annotations \
  --resume
```

标注规范见 `attachments/labeling_guide.md`。

脚本操作：

| 操作 | 作用 |
| --- | --- |
| 鼠标左键 | 保存当前目标点为正样本 |
| `N` | 保存为明确无可靠巡线目标的负样本 |
| `X` | 标记为拒绝样本，不进入训练 |
| `A` / `D` | 上一张 / 下一张 |
| `Z` / `C` | 向前 / 向后跳 5 张 |
| `Q` | 保存现有标注并退出 |

每张图片对应一个同名 TXT：

```text
正样本：x_roi y_roi 1
负样本：nan nan 0
拒绝样本：nan nan -1
```

坐标保存为 `640×224` ROI 中的原始像素坐标。归一化由后续数据集加载器统一
完成，标注脚本不写 `[0,1]` 或 `[-1,1]` 坐标。

#### E3. 试标验收

- 100 张图片全部具有明确状态。
- 正样本 TXT 为三个字段，第三个字段为 `1`。
- 负样本 TXT 为 `nan nan 0`。
- 拒绝样本 TXT 为 `nan nan -1`。
- 至少 20 张由第二人独立复标。
- 生成包含图片、目标点和坐标的可视化拼图。
- 坐标全部落在 `640×224` ROI 内。

### 阶段 F：正式数据集

Manifest 可以理解为数据集的总账或索引：它记录每张图片来自哪个视频和帧、
标签是什么、审核状态如何，以及属于哪个 split。只依赖文件名和文件夹无法完整
表达采集场景、标注状态和划分依据，也难以检查数据泄漏与复现实验。

#### F1. 数据规模

最低要求：

| 类别 | 数量或覆盖要求 |
| --- | --- |
| 正样本 | 不少于 1,500 张 |
| 真实负样本 | 不少于 200 张 |
| 独立视频 | 不少于 6 段 |
| 光照 | 至少两种条件 |
| 驾驶偏差 | 居中、左偏、右偏均覆盖 |
| 二次复核 | 不少于正式样本的 10% |

数据充足时，建议达到 3,000～6,000 张正样本，真实负样本占
10%～20%。不要用连续重复帧凑数量。

#### F2. 统一标签

每个样本在 `dataset_manifest.csv` 中至少记录：

```text
sample_id
video_id
frame_index
timestamp_ms
image_path
task
lap
direction
lighting
x_roi
y_roi
x_norm
y_norm
has_track
annotation_status
reviewer
split
```

状态含义：

| 状态 | 含义 |
| --- | --- |
| `unlabeled` | 尚未完成标注 |
| `positive` | 存在可靠 `track_center` |
| `negative` | 已确认没有可靠目标点 |
| `rejected` | 图片质量或内容无效 |
| `reviewed` | 标注已经复核 |

缺少同名 TXT 只能视为 `unlabeled`，不能自动变成 `negative`。

#### F3. 数据集验证

使用附件将帧清单和 TXT 标签合并为统一数据集清单：

```bash
python "$TRAIN_ROOT/validate_dataset.py" \
  --frame-manifest Task2/data/frame_manifest.csv \
  --video-manifest Task2/data/video_manifest.csv \
  --output Task2/data/dataset_manifest.csv \
  --require-complete

python "$TRAIN_ROOT/render_annotations.py" \
  --manifest Task2/data/dataset_manifest.csv \
  --output Task2/output/annotation_contact_sheet.jpg
```

`validate_dataset.py` 和人工检查必须确认：

- 每张正式样本都有来源。
- 正样本恰好有一个目标点。
- 负样本不包含伪坐标。
- 坐标位于 ROI 范围内。
- 正归一化和反归一化能够往返。
- 图片和标签一一对应。
- 没有重复 `sample_id`。
- 没有 NaN、Inf 或空图片。
- 所有标注都能渲染回原图。

### 阶段 G：按视频划分数据集

数据泄漏是指测试集通过重复或高度相似的数据，提前暴露了训练集的信息。同一段
视频中的相邻帧通常只有很小差异；如果按图片随机划分，它们可能分别进入训练集
和测试集，使测试指标看起来很好，却不能代表模型面对新视频时的表现。因此本
任务必须按完整视频或连续录制场次划分。

默认比例：

```text
train 70%
val   15%
test  15%
```

划分单位为 `video_id`、完整圈次或连续录制场次，不是单张图片。

要求：

- 同一视频只能属于一个集合。
- 同一连续录制场次不能跨集合。
- 跨集合近似重复帧数量为 0。
- 三个集合尽量覆盖方向、光照和路线。
- test 在第一次正式训练前冻结。
- 训练期间不得根据 test 指标选择 epoch 或调整超参数。

划分只写入 `split_manifest.csv`，不依靠随机移动图片表示。

使用附件按 `video_id` 写入划分：

```bash
python "$TRAIN_ROOT/split_dataset.py" \
  --manifest Task2/data/dataset_manifest.csv \
  --split-manifest Task2/data/split_manifest.csv
```

首次划分后应人工检查场景覆盖并冻结 test。若需要重新生成，必须明确使用
`--force`，并记录重新划分的原因。

### 阶段 H：两输出教学基线

两输出模型只学习正样本的 x、y 坐标，用较少的变量检查图片、标签、预处理、
损失和训练循环是否连通。再用 20～50 张图片做小数据过拟合测试：如果模型连
这批小数据都无法记住，通常说明管线存在问题，此时不应直接扩大数据量或延长
正式训练。

附件中的 `train.py --baseline` 会建立最小教学基线：

```text
backbone: ResNet18
pretrained: ImageNet
input: 1×3×224×224
output: [x_norm, y_norm]
samples: positive only
loss: SmoothL1
```

先用 20～50 张图片进行过拟合测试，基线应该能够明显记住这批小数据。

如果小数据集无法过拟合，优先排查：

- 图片和标签是否对应。
- ROI 是否一致。
- RGB/BGR 是否错误。
- 坐标是否重复归一化。
- y 轴是否意外翻转。
- 模型参数是否真正更新。

两输出基线用于教学和管线检查，不作为最终交付模型。

运行示例：

```bash
python "$TRAIN_ROOT/train.py" \
  --config "$TRAIN_ROOT/configs/train_resnet18_xyv.yaml" \
  --manifest Task2/data/dataset_manifest.csv \
  --baseline
```

### 阶段 I：三输出最终模型

三输出模型同时解决两个问题：回归预测连续的 x、y 坐标，分类判断当前画面是否
存在可靠轨迹。`track_logit` 是 sigmoid 之前的原始分类输出；mask loss 表示
坐标损失只在正样本上计算。一个 epoch 是模型完整遍历一次训练集，batch size
是每次参数更新共同处理的样本数。

#### I1. 模型

```text
backbone: torchvision ResNet18
pretrained: ImageNet
head: Linear(512, 3)
output: [x_norm, y_norm, track_logit]
```

模型内不对 `track_logit` 执行 sigmoid。评估和部署后处理使用：

```text
confidence = sigmoid(track_logit)
```

#### I2. 损失

坐标损失仅对 `has_track=1` 的样本计算：

```text
coordinate_loss = SmoothL1(pred_xy, target_xy)
confidence_loss = BCEWithLogits(pred_logit, has_track)
total_loss = coordinate_loss(valid samples only)
             + λ × confidence_loss
```

默认 `λ=1.0`。训练日志必须分别记录坐标损失和置信度损失，不能只记录总损失。

#### I3. 默认训练参数

以附件中的 `configs/train_resnet18_xyv.yaml` 为准，当前配置为：

```text
seed: 42
batch_size: 32
epochs: 100
optimizer: AdamW
learning_rate: 0.0003
weight_decay: 0.0001
early_stopping_patience: 15
```

#### I4. 数据增强

训练集当前使用：

- `ColorJitter`：亮度、对比度和饱和度变化范围均为 `0.2`，色相变化范围为
  `0.05`。
- 概率为 `0.1` 的 `GaussianBlur`。

当前不使用水平翻转、垂直翻转、旋转或随机裁剪。验证集和测试集只执行固定的
`Resize`、`ToTensor` 和 ImageNet mean/std 归一化，不使用随机数据增强。

#### I5. 训练输出

```text
runs/<run_id>/
├── config.yaml
├── metrics.csv
├── tensorboard/
├── best_model.pth
├── last_model.pth
└── run_metadata.json
```

`run_metadata.json` 至少包含：

- Git commit。
- 数据集版本或 manifest SHA256。
- 随机种子。
- Python、PyTorch 和 torchvision 版本。
- CPU/GPU 信息。
- 开始和结束时间。
- 最佳 epoch 和选择依据。

运行三输出最终模型训练：

```bash
python "$TRAIN_ROOT/train.py" \
  --config "$TRAIN_ROOT/configs/train_resnet18_xyv.yaml" \
  --manifest Task2/data/dataset_manifest.csv
```

### 阶段 J：模型评估

MAE 回答“平均误差有多大”；P90 回答“90% 样本的误差不超过多少”，更能反映
尾部失败。Precision 关注预测为有轨迹时有多少是真的，用来观察误检；Recall
关注真实有轨迹时找回了多少，用来观察漏检；F1 综合平衡 Precision 和 Recall。
这些指标必须与最大误差图片和分场景结果一起分析。

#### J1. 坐标指标

仅对 `has_track=1` 的样本计算：

- x 方向 MAE，单位像素。
- y 方向 MAE，单位像素。
- 二维点距离 MAE。
- 二维点距离 P90 和 P95。
- 误差小于 10、20、30 像素的比例。

#### J2. 置信度指标

- Precision。
- Recall。
- F1。
- 混淆矩阵。
- 不同阈值下的 Precision/Recall。

#### J3. 分场景报告

至少按以下维度拆分：

- 直线、左弯、右弯。
- 顺时针、逆时针。
- 正常、偏暗、局部强光。
- 居中、左偏、右偏。
- 不同 `video_id`。

#### J4. 失败案例

当前 `evaluate.py` 保存定量结果、逐样本预测和最大误差图片，
`infer_video.py` 保存独立视频的预测叠加结果：

```text
Task2/output/
├── evaluation/
│   ├── metrics.json
│   ├── predictions.csv
│   └── largest_errors/
├── test_video.mp4
└── test_video.json
```

最大误差图片叠加预测点、人工真值点和像素误差。独立测试视频叠加：

- 预测点。
- 置信度。
- 帧号。

当前附件的评估与独立视频推理命令：

```bash
python "$TRAIN_ROOT/evaluate.py" \
  --checkpoint Task2/runs/<run_id>/best_model.pth \
  --manifest Task2/data/dataset_manifest.csv \
  --split test \
  --output-dir Task2/output/evaluation

python "$TRAIN_ROOT/infer_video.py" \
  --checkpoint Task2/runs/<run_id>/best_model.pth \
  --video Task2/data/raw/<independent_test_video>.mp4 \
  --output Task2/output/test_video.mp4
```

### 阶段 K：ONNX 导出与一致性

ONNX 是不同训练和推理工具之间交换模型的格式。PyTorch 与 ONNX Runtime
可能使用不同的算子实现和浮点计算顺序，因此输出不要求逐位相同，只要求误差
落在约定范围内。Task2 负责导出浮点 ONNX 并验证转换没有明显改变结果；Task3
再使用 OpenExplorer 进行 PTQ、编译和板端部署，并单独评估量化误差。

#### K1. 导出契约

```text
input name: input
input shape: [1, 3, 224, 224]
input dtype: float32
output name: output
output shape: [1, 3]
output dtype: float32
opset: 11
dynamic axes: false
```

固定 batch 和尺寸，便于 Task3 进行 OpenExplorer 检查和转换。

使用当前附件导出：

```bash
python "$TRAIN_ROOT/export_onnx.py" \
  --checkpoint Task2/runs/<run_id>/best_model.pth \
  --output Task2/output/line_follower_resnet18.onnx
```

#### K2. 一致性验证

从冻结 test 集选择至少 100 张图片，对相同的预处理张量分别执行：

```text
PyTorch → raw output
ONNX Runtime → raw output
```

比较：

- 三个原始输出值。
- 反归一化后的 ROI 像素坐标。
- sigmoid 后的置信度。
- 是否出现 NaN 或 Inf。

使用当前附件对 100 张冻结 test 图片进行一致性验证：

```bash
python "$TRAIN_ROOT/verify_onnx.py" \
  --checkpoint Task2/runs/<run_id>/best_model.pth \
  --onnx Task2/output/line_follower_resnet18.onnx \
  --manifest Task2/data/dataset_manifest.csv \
  --samples 100 \
  --output Task2/output/onnx_parity_report.json
```

通过条件：

- `onnx.checker` 通过。
- ONNX Runtime 能加载并完成推理。
- 输入输出名称、形状和类型与契约一致。
- 100 张图片全部完成推理。
- 最大输出绝对差不超过 `1e-4`。
- 反归一化坐标差不超过 `0.1 px`。
- NaN 和 Inf 数量为 0。

## 7. 验收与提交

### 7.1 模型性能指标

数据工程和 ONNX 一致性属于硬门槛。下表作为模型性能验收指标，冻结 test 后
不得通过重新划分或更换 test 集降低难度。

验收值：

| 指标 | 通过值 |
| --- | ---: |
| 可见目标点 x MAE | ≤ 20 px |
| 二维点距离 P90 | ≤ 35 px |
| 置信度 F1 | ≥ 0.90 |
| 20 px 内命中率 | ≥ 80% |
| 推理 NaN/Inf | 0 |
| PyTorch/ONNX 坐标差 | ≤ 0.1 px |

### 7.2 验收标准

| 类别 | 验收项 | 通过条件 |
| --- | --- | --- |
| 环境 | WSL2 | Ubuntu 22.04 运行在 WSL2 |
| 环境 | Conda | `drobot-train` 可从 `environment.yml` 重新创建 |
| 环境 | Python | 环境内 Python 为 3.10.x，且不是系统 Python 或 `base` |
| 环境 | ROS 2 | 不安装、不作为 Task2 依赖 |
| 输入 | Task1 数据 | MP4、JSON 和清单完整匹配 |
| 追溯 | 视频与帧 | 每张图片可反查视频、帧号和时间 |
| ROI | 裁剪 | 统一使用 `frame[256:480, 0:640]` |
| 标注 | 标签完整性 | 正样本一个点，负样本显式标记 |
| 标注 | 坐标 | 无越界、NaN、Inf 或重复归一化 |
| 划分 | 分组 | 同一视频不跨 train/val/test |
| 泄漏 | 重复帧 | 跨集合近似重复帧为 0 |
| 训练 | 可复现 | 配置、seed、版本和日志完整 |
| 评估 | 独立 test | 有定量指标、分场景结果和失败案例 |
| ONNX | 模型契约 | 输入 `[1,3,224,224]`，输出 `[1,3]` |
| ONNX | 一致性 | PyTorch 与 ONNX Runtime 满足误差门槛 |
| 提交 | Task3 交接 | 模型、契约、配置、哈希和报告齐全 |

### 7.3 学习总结与思考

完成任务后，使用 `templates/acceptance_report.md` 中的“学习总结与思考”回答
全部问题。答案需要结合自己的 manifest、训练日志、评估指标、ONNX 一致性报告
和失败案例图片，不能只抄写本文定义。

## 8. 实施里程碑

| 里程碑 | 内容 | 退出条件 |
| --- | --- | --- |
| M1 | WSL2 和 Conda 训练环境 | 环境报告通过 |
| M2 | 数据与模型契约 | `model_contract.yaml` 与数据往返验证通过 |
| M3 | 视频导入与抽帧 | 50 张 ROI 人工检查通过 |
| M4 | 100 张试标 | 标注规范和坐标可视化通过 |
| M5 | 两输出基线 | 小数据过拟合和独立推理通过 |
| M6 | 数据集 v1 | manifest、验证、视频级 split 通过 |
| M7 | 三输出模型 | 正式训练和 test 评估完成 |
| M8 | ONNX 交付 | 一致性报告和 Task3 模型包完成 |

## 9. 当前附件脚本

以下文件现已位于
`Task2/attachments/track_detection_resnet18/`，并随 Task2 交付。

| 顺序 | 当前文件 | 作用 |
| ---: | --- | --- |
| 1 | `check_environment.py` | 检查 Conda、PyTorch、OpenCV、ONNX 并生成环境报告 |
| 2 | `import_task1_data.py` | 校验 MP4、JSON 和视频清单 |
| 3 | `extract_frames.py` | 按时间抽帧、统一尺寸、裁剪 ROI 和质量标记 |
| 4 | `label_image.py` | 点击标注正样本，并支持负样本和拒绝样本 |
| 5 | `validate_dataset.py` | 校验 TXT 标签并生成统一数据集清单 |
| 6 | `render_annotations.py` | 生成随机抽样的标注可视化拼图 |
| 7 | `split_dataset.py` | 按 `video_id` 写入 train/val/test 划分 |
| 8 | `train.py` | 训练两输出教学基线和三输出最终模型 |
| 9 | `evaluate.py` | 计算图片级指标并输出最大误差案例 |
| 10 | `infer_video.py` | 对独立视频推理并生成叠加结果 |
| 11 | `export_onnx.py` | 导出并检查固定输入 ONNX |
| 12 | `verify_onnx.py` | 比较 PyTorch 与 ONNX Runtime 输出 |
| 13 | `common.py` | 路径、YAML、哈希、JSON 和随机种子公共函数 |
| 14 | `dataset.py` | Manifest 数据集和训练/评估预处理 |
| 15 | `model.py` | ResNet18 构建和 checkpoint 读取 |
| 16 | `metrics.py` | 坐标像素误差与置信度指标 |

附件只包含 Task2 的数据处理、浮点训练、评估和 ONNX 交付代码。
OpenExplorer 检查、PTQ、编译和板端推理文件不放入该目录，由 Task3 提供。

## 10. 相关资料

- [Microsoft：安装 WSL](https://learn.microsoft.com/windows/wsl/install)
- [Microsoft：WSL 基本命令](https://learn.microsoft.com/windows/wsl/basic-commands)
- [Microsoft：在 WSL 中启用 NVIDIA CUDA](https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl)
- [Anaconda：安装 Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install)
- [Conda：管理和创建环境](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)
- [PyTorch：Start Locally](https://pytorch.org/get-started/locally/)
