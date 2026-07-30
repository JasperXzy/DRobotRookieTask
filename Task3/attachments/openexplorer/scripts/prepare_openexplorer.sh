#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
version="1.2.8"
base_url="https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/${version}"
download_dir="${OPENEXPLORER_DOWNLOAD_DIR:-${repo_root}/Downloads}"
sdk_archive="${download_dir}/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz"
doc_archive="${download_dir}/x5_doc-v1.2.8-py310-cn.zip"
docker_archive="${download_dir}/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz"
release_note="${download_dir}/release_note_CN.txt"
sdk_root="${HOME}/open_explorer"
image="${OPENEXPLORER_IMAGE:-openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310}"

for required_command in docker stat tar wget; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "Required command is unavailable: ${required_command}" >&2
    exit 1
  fi
done

mkdir -p "${download_dir}"

download_or_resume() {
  local url="$1"
  local output="$2"
  local expected_bytes="$3"
  local actual_bytes="0"

  if [[ -f "${output}" ]]; then
    actual_bytes="$(stat -c '%s' "${output}")"
  fi

  if [[ "${actual_bytes}" == "${expected_bytes}" ]]; then
    echo "Using existing file: ${output} (${actual_bytes} bytes)"
    return
  fi

  if (( actual_bytes > expected_bytes )); then
    echo "Existing file is larger than expected: ${output}" >&2
    echo "Expected ${expected_bytes} bytes, found ${actual_bytes} bytes." >&2
    exit 1
  fi

  echo "Downloading or resuming: ${output} (${actual_bytes}/${expected_bytes} bytes)"
  wget -c "${url}" -O "${output}"
  actual_bytes="$(stat -c '%s' "${output}")"
  if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
    echo "Downloaded file size mismatch: ${output}" >&2
    echo "Expected ${expected_bytes} bytes, found ${actual_bytes} bytes." >&2
    exit 1
  fi
}

download_or_resume \
  "${base_url}/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz" \
  "${sdk_archive}" \
  "2280017920"
download_or_resume \
  "${base_url}/x5_doc-v1.2.8-py310-cn.zip" \
  "${doc_archive}" \
  "20838748"
download_or_resume \
  "${base_url}/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz" \
  "${docker_archive}" \
  "1815507981"
download_or_resume \
  "${base_url}/release_note_CN.txt" \
  "${release_note}" \
  "1149"

if [[ ! -d "${sdk_root}" ]]; then
  mkdir -p "${sdk_root}"
  # The SDK archive keeps a .tar.gz suffix but may be an uncompressed POSIX tar.
  tar -xf "${sdk_archive}" -C "${sdk_root}" --strip-components=1
else
  echo "SDK directory already exists; extraction skipped: ${sdk_root}"
fi

if docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Docker image already loaded: ${image}"
else
  docker load -i "${docker_archive}"
fi

docker image inspect "${image}" --format '{{.RepoTags}} {{.Id}}'
echo "OpenExplorer SDK root: ${sdk_root}"
echo "Documentation archive: ${doc_archive}"
echo "Release notes: ${release_note}"
