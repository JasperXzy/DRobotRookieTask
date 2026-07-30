#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script must run inside the Ubuntu WSL distribution." >&2
  exit 1
fi

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as the normal WSL user; it invokes sudo when needed." >&2
  exit 1
fi

if ! command -v systemctl >/dev/null 2>&1 || [[ "$(ps -p 1 -o comm=)" != "systemd" ]]; then
  echo "WSL systemd is not active. Add the following to /etc/wsl.conf, then run 'wsl --shutdown' from Windows:" >&2
  echo "[boot]" >&2
  echo "systemd=true" >&2
  exit 1
fi

if command -v docker >/dev/null 2>&1; then
  docker --version
  echo "Docker is already installed; no changes were made."
  exit 0
fi

for conflicting_package in docker.io docker-compose docker-compose-v2 podman-docker containerd runc; do
  if dpkg-query -W -f='${Status}' "${conflicting_package}" 2>/dev/null | grep -q "install ok installed"; then
    echo "Conflicting package detected: ${conflicting_package}" >&2
    echo "Review and remove conflicting packages manually before rerunning this script." >&2
    exit 1
  fi
done

sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

ubuntu_codename="$(. /etc/os-release && printf '%s' "${VERSION_CODENAME}")"
ubuntu_arch="$(dpkg --print-architecture)"
docker_source="deb [arch=${ubuntu_arch} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${ubuntu_codename} stable"
printf '%s\n' "${docker_source}" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "${USER}"

sudo docker run --rm hello-world

echo
echo "Docker CE is installed and running in WSL."
echo "Close this WSL shell and reconnect so the docker group membership takes effect."
echo "Keep Docker Desktop integration disabled for this distribution to avoid using two Docker daemons."
