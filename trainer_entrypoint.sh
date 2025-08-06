#!/bin/bash
set -e

if ! command -v nvcc &> /dev/null; then
  echo "CUDA Toolkit이 설치되어 있지 않습니다. 설치를 시작합니다..."
  wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-rhel7-12-4-local-12.4.0_550.54.14-1.x86_64.rpm
  sudo rpm -i cuda-repo-rhel7-12-4-local-12.4.0_550.54.14-1.x86_64.rpm
  sudo yum clean all
  sudo yum -y install cuda-toolkit-12-4
fi

if ! command -v git &> /dev/null; then
  echo "Git이 설치되어 있지 않습니다. 설치를 시작합니다..."
  apt-get update && apt-get install -y git
fi

REPO_URL="https://github.com/X2bee/praque_trainer.git"

if [ ! -d "/app/lm-evaluation-harness" ]; then
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness /app/lm-evaluation-harness
  cd /app/lm-evaluation-harness
  pip install -e .
  cd ..
else
  echo "lm-evaluation-harness 디렉터리가 존재하므로 업데이트(pull)합니다..."
  cd /app/lm-evaluation-harness
  git pull
  cd ..
fi

if [ ! -d "/app/trainer" ]; then
  echo "trainer 디렉터리가 없으므로 클론합니다..."
  git clone -b main --single-branch "${REPO_URL}" /app/trainer
else
  echo "trainer 디렉터리가 존재하므로 업데이트(pull)합니다..."
  cd /app/trainer
  # 원격 저장소 URL에 인증 정보 업데이트
  git remote set-url origin "${REPO_URL}"
  git pull
  cd ..
fi

echo "필수 패키지를 설치합니다..."
pip install -r /app/trainer/requirements.txt

echo "Run Trainer - main.py"
python /app/trainer/main.py
