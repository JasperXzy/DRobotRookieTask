#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
task3_root="${repo_root}/Task3"
model_dir="${task3_root}/output/model_output"
bin_model="${model_dir}/race_track_detection_224x224_nv12.bin"
log_dir="${task3_root}/output/logs"

[[ -s "${bin_model}" ]] || { echo "Missing BIN model: ${bin_model}" >&2; exit 1; }
mkdir -p "${log_dir}"

hb_model_info "${bin_model}" 2>&1 | tee "${log_dir}/hb_model_info.log"
cd "${task3_root}/output"
hb_perf "${bin_model}" 2>&1 | tee "${log_dir}/hb_perf.log"
sha256sum "${bin_model}" | tee "${task3_root}/output/model_sha256.txt"
