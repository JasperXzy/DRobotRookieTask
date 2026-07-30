#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
sdk_root="${OPENEXPLORER_SDK_ROOT:-${HOME}/open_explorer}"
image="${OPENEXPLORER_IMAGE:-openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310}"

if [[ ! -d "${sdk_root}" ]]; then
  echo "OpenExplorer SDK directory not found: ${sdk_root}" >&2
  echo "Run prepare_openexplorer.sh first, or set OPENEXPLORER_SDK_ROOT." >&2
  exit 1
fi

if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Docker image not found: ${image}" >&2
  echo "Run prepare_openexplorer.sh first." >&2
  exit 1
fi

docker_args=(
  run
  --rm
  -it
  --shm-size=4g
  --workdir=/workspace
  --volume="${repo_root}:/workspace"
  --volume="${sdk_root}:/open_explorer"
  "${image}"
)

if [[ "$#" -eq 0 ]]; then
  exec docker "${docker_args[@]}" bash
fi

exec docker "${docker_args[@]}" "$@"
