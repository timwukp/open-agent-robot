#!/usr/bin/env bash
# open-agent-robot — EC2 software setup (Ubuntu 24.04, g4dn / T4).
# Run ON the instance. Stages so a reboot can happen after the driver:
#   ./setup-ec2.sh driver   (installs NVIDIA driver; reboot afterwards)
#   ./setup-ec2.sh stack    (ROS 2 Jazzy + Python env; idempotent)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

stage_driver() {
  sudo apt-get update -q
  sudo apt-get install -yq ubuntu-drivers-common build-essential
  sudo ubuntu-drivers install --gpgpu
  # --gpgpu installs only the headless kernel/compute stack; add the user-space
  # tools (nvidia-smi) and GL/EGL libs MuJoCo offscreen rendering needs.
  DRVVER=$(dpkg -l | grep -oE 'nvidia-headless-no-dkms-[0-9]+-server' | grep -oE '[0-9]+' | head -1)
  sudo apt-get install -yq "nvidia-utils-${DRVVER}-server" "libnvidia-gl-${DRVVER}-server"
  echo "driver installed — reboot now, then run: ./setup-ec2.sh stack"
}

stage_stack() {
  # --- ROS 2 Jazzy (Ubuntu 24.04) ---
  sudo apt-get update -q
  sudo apt-get install -yq curl gnupg lsb-release software-properties-common
  sudo add-apt-repository -y universe
  if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
      | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
    sudo apt-get update -q
  fi
  sudo apt-get install -yq ros-jazzy-ros-base ros-jazzy-vision-msgs \
    ros-jazzy-foxglove-bridge ros-jazzy-cv-bridge python3-colcon-common-extensions
  grep -q 'source /opt/ros/jazzy' ~/.bashrc || echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc

  # --- system deps for rendering/audio ---
  sudo apt-get install -yq libegl1 libgl1 libgles2 libglib2.0-0 ffmpeg python3-venv python3-dev

  # --- Python env (kept separate from ROS's system python; nodes source both) ---
  python3 -m venv ~/venv --system-site-packages
  ~/venv/bin/pip install -q --upgrade pip
  ~/venv/bin/pip install -q \
    "mujoco>=3.2" \
    "ultralytics" \
    "tensorrt" \
    "funasr" "modelscope" "torch" "torchaudio" --extra-index-url https://download.pytorch.org/whl/cu124
  ~/venv/bin/pip install -q "kokoro>=0.9" "soundfile" "misaki[zh]"
  ~/venv/bin/pip install -q mediapipe || echo "WARN: mediapipe install failed (fallback: YOLO26-pose per plan)"
  ~/venv/bin/pip install -q websockets numpy opencv-python-headless
  # funasr pins an old boto3 (no converse_stream) and a broken coverage; fix last
  ~/venv/bin/pip install -qU boto3 botocore "coverage>=7.6"
  echo "stack done"
}

case "${1:-}" in
  driver) stage_driver ;;
  stack)  stage_stack ;;
  *) echo "usage: $0 {driver|stack}"; exit 1 ;;
esac
