#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
task3_root="${repo_root}/Task3"
config="${task3_root}/attachments/openexplorer/config/resnet18_xyv.yaml"
onnx_model="${repo_root}/Task2/output/line_follower_resnet18.onnx"
calibration_dir="${task3_root}/output/calibration_data_rgb_f32"
log_dir="${task3_root}/output/logs"

[[ -s "${onnx_model}" ]] || { echo "Missing ONNX model: ${onnx_model}" >&2; exit 1; }
[[ -d "${calibration_dir}" ]] || { echo "Missing calibration directory: ${calibration_dir}" >&2; exit 1; }
find "${calibration_dir}" -maxdepth 1 -type f -name '*.rgb' -print -quit | grep -q . || {
  echo "No calibration .rgb files found in ${calibration_dir}" >&2
  exit 1
}

mkdir -p "${log_dir}"
cd "${task3_root}/output"

hb_mapper makertbin \
  --config "${config}" \
  --model-type onnx \
  2>&1 | tee "${log_dir}/hb_mapper_makertbin.log"
