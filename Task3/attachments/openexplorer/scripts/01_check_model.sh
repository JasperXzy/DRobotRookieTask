#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
task3_root="${repo_root}/Task3"
config="${task3_root}/attachments/openexplorer/config/resnet18_xyv.yaml"
onnx_model="${repo_root}/Task2/output/line_follower_resnet18.onnx"
output_dir="${task3_root}/output"
log_dir="${output_dir}/logs"

mkdir -p "${log_dir}"
cd "${repo_root}"

python3 Task3/attachments/openexplorer/check_environment.py \
  --onnx "${onnx_model}" \
  --config "${config}" \
  --output "${output_dir}/environment_report.json"

cd "${log_dir}"
hb_mapper checker \
  --model-type onnx \
  --model "${onnx_model}" \
  --march bayes-e \
  2>&1 | tee "${log_dir}/hb_mapper_checker.log"
