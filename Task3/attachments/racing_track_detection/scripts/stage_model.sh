#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd -- "${script_dir}/.." && pwd)"
repo_root="$(cd -- "${package_root}/../../.." && pwd)"
source_model="${repo_root}/Task3/output/model_output/race_track_detection_224x224_nv12.bin"
target_model="${package_root}/config/race_track_detection_224x224_nv12.bin"

[[ -s "${source_model}" ]] || { echo "Missing generated model: ${source_model}" >&2; exit 1; }
if [[ -e "${target_model}" && "${1:-}" != "--force" ]]; then
  echo "Target already exists: ${target_model}" >&2
  echo "Pass --force only when deliberately replacing the staged model." >&2
  exit 1
fi

cp "${source_model}" "${target_model}"
sha256sum "${source_model}" "${target_model}"
echo "Model staged for colcon build: ${target_model}"
