#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
model_dir="${repo_root}/Task3/output/model_output"
quantized_model="${model_dir}/race_track_detection_224x224_nv12_quantized_model.onnx"
bin_model="${model_dir}/race_track_detection_224x224_nv12.bin"
log_dir="${repo_root}/Task3/output/logs"
hrt_tools="/open_explorer/package/host/hrt_tools"
dnn_lib="/open_explorer/samples/ai_toolchain/horizon_runtime_sample/code/deps_gcc11.3/x86/dnn_x86/lib"

[[ -s "${quantized_model}" ]] || { echo "Missing quantized ONNX: ${quantized_model}" >&2; exit 1; }
[[ -s "${bin_model}" ]] || { echo "Missing BIN model: ${bin_model}" >&2; exit 1; }
[[ -x "${hrt_tools}/hrt_model_exec" ]] || {
  echo "Missing x86 hrt_model_exec: ${hrt_tools}/hrt_model_exec" >&2
  exit 1
}
[[ -e "${dnn_lib}/libdnn.so" ]] || {
  echo "Missing x86 libdnn.so: ${dnn_lib}/libdnn.so" >&2
  exit 1
}

export PATH="${hrt_tools}:${PATH}"
export LD_LIBRARY_PATH="${dnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${log_dir}"
cd "${repo_root}/Task3/output"

hb_verifier \
  -m "${quantized_model},${bin_model}" \
  -s True \
  2>&1 | tee "${log_dir}/hb_verifier.log"
