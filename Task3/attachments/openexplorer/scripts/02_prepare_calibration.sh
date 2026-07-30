#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"

cd "${repo_root}"
python3 Task3/attachments/openexplorer/prepare_calibration.py \
  --manifest Task2/data/dataset_manifest.csv \
  --output-dir Task3/output/calibration_data_rgb_f32 \
  --selection-manifest Task3/output/calibration_manifest.csv \
  --report Task3/output/calibration_report.json \
  --count 64 \
  --seed 42 \
  --splits train val
