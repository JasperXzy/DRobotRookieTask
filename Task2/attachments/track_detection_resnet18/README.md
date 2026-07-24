# Task2 ResNet18 巡线训练工具包

本目录是 Task2 随任务提供的自包含可执行代码。

从仓库根目录运行脚本。推荐先定义：

```bash
TRAIN_ROOT=Task2/attachments/track_detection_resnet18
conda activate drobot-train
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `common.py` | 路径、YAML、哈希、JSON 和随机种子工具 |
| `dataset.py` | 从 `dataset_manifest.csv` 读取数据并执行固定预处理 |
| `model.py` | 创建 ResNet18 两输出/三输出模型并读取 checkpoint |
| `metrics.py` | 像素坐标误差和置信度指标 |
| `configs/dataset.yaml` | 数据路径、抽帧、ROI、标签和划分配置 |
| `configs/train_resnet18_xyv.yaml` | ResNet18 训练、增强、评估和导出配置 |
| `configs/model_contract.yaml` | 与 Task3 共享的模型输入输出契约 |
| `requirements.txt` | 附件通用 Python 依赖；PyTorch 单独按设备安装 |
| `check_environment.py` | 生成训练环境报告 |
| `import_task1_data.py` | 校验 Task1 MP4/JSON 并生成视频清单 |
| `extract_frames.py` | 按时间抽帧、缩放、裁剪 ROI 并生成帧清单 |
| `label_image.py` | 交互式目标点、负样本和拒绝样本标注 |
| `validate_dataset.py` | 校验标签并生成统一数据集清单 |
| `render_annotations.py` | 输出标注可视化拼图 |
| `split_dataset.py` | 按 `video_id` 生成 train/val/test 划分 |
| `train.py` | 两输出基线和三输出最终模型训练 |
| `evaluate.py` | 冻结 split 的定量评估与逐样本预测 |
| `infer_video.py` | 对独立 MP4 执行推理并输出叠加视频 |
| `export_onnx.py` | 导出固定 `[1,3,224,224]` ONNX |
| `verify_onnx.py` | 验证 PyTorch 与 ONNX Runtime 输出一致性 |

Task2 只覆盖浮点训练和 ONNX 交付。OpenExplorer、PTQ、模型编译和板端
推理由 Task3 的独立环境和脚本完成。
