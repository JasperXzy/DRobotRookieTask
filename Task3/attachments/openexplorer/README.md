# Task3 OpenExplorer 工具包

本目录用于在 WSL2 Ubuntu x86-64 中，通过 X5 OpenExplorer v1.2.8 CPU
Docker 镜像完成模型检查、校准数据准备、PTQ、量化 ONNX 评估和 BIN 一致性
验证。OpenExplorer 在 Windows 的 WSL2 Ubuntu 中运行。

所有 OpenExplorer 容器内命令都从仓库根目录 `/workspace` 执行。

## 文件

| 文件 | 用途 |
| --- | --- |
| `config/resnet18_xyv.yaml` | X5 `bayes-e`、NV12 运行时输入的 PTQ 配置 |
| `check_environment.py` | 检查平台、工具版本、ONNX 契约与 YAML |
| `prepare_calibration.py` | 从 train/val 选择代表性样本并生成 RGB CHW float32 |
| `evaluate_quantized.py` | 同图比较浮点 ONNX 与量化 ONNX |
| `scripts/install_docker_wsl.sh` | 在启用 systemd 的 Ubuntu WSL 中安装 Docker CE |
| `scripts/prepare_openexplorer.sh` | 下载/复用 v1.2.8 发布物、解包 SDK、加载离线镜像 |
| `scripts/enter_openexplorer.sh` | 把仓库挂载为 `/workspace` 并进入 CPU 容器 |
| `scripts/01_check_model.sh` | 环境、模型契约与 `hb_mapper checker` |
| `scripts/02_prepare_calibration.sh` | 生成 64 个校准二进制 |
| `scripts/03_build_model.sh` | `hb_mapper makertbin` |
| `scripts/04_inspect_bin.sh` | `hb_model_info`、`hb_perf` 与 SHA256 |
| `scripts/05_evaluate_quantized.sh` | 正式 test 量化精度验收 |
| `scripts/06_verify_bin.sh` | `hb_verifier` 检查量化 ONNX/BIN 一致性 |

## 标准顺序

四个 X5 v1.2.8 发布物统一存放在仓库根目录 `Downloads/`。
`prepare_openexplorer.sh` 会自动定位该目录，复用字节数正确的已有文件，并对缺失或
未下载完整的文件执行断点续传。
离线包加载后的固定镜像标签为
`openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310`。

在 WSL 宿主执行：

```bash
cd ~/DRobotRookieTask
bash Task3/attachments/openexplorer/scripts/install_docker_wsl.sh
```

重新连接 WSL，使 Docker group 生效，然后：

```bash
cd ~/DRobotRookieTask
bash Task3/attachments/openexplorer/scripts/prepare_openexplorer.sh
bash Task3/attachments/openexplorer/scripts/enter_openexplorer.sh
```

进入容器后：

```bash
bash Task3/attachments/openexplorer/scripts/01_check_model.sh
bash Task3/attachments/openexplorer/scripts/02_prepare_calibration.sh
bash Task3/attachments/openexplorer/scripts/03_build_model.sh
bash Task3/attachments/openexplorer/scripts/04_inspect_bin.sh
bash Task3/attachments/openexplorer/scripts/05_evaluate_quantized.sh
bash Task3/attachments/openexplorer/scripts/06_verify_bin.sh
```

## 数据契约

校准文件不是 JPEG，也不是归一化后的 Tensor。每个 `.rgb` 文件必须是：

```text
color: RGB
layout: CHW
shape: [3, 224, 224]
dtype: float32
range: [0, 255]
size: 602112 bytes
normalization: none
```

mean 和 scale 由 `resnet18_xyv.yaml` 注入。校准脚本只使用 train/val，禁止从
冻结 test 集挑选校准样本。

## 最终验收门禁

量化评估只提供 `final_acceptance` 模式。冻结 test 必须同时包含正样本和负样本；
否则 `negative_samples_present` 检查失败，需要返回 Task2 补标、重训并重新导出
ONNX。
