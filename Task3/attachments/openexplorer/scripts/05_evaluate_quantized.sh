#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
model_dir="${repo_root}/Task3/output/model_output"

cd "${repo_root}"
python3 Task3/attachments/openexplorer/evaluate_quantized.py \
  --float-model Task2/output/line_follower_resnet18.onnx \
  --quantized-model "${model_dir}/race_track_detection_224x224_nv12_quantized_model.onnx" \
  --manifest Task2/data/dataset_manifest.csv \
  --split test \
  --output-dir Task3/output/quantized_evaluation
