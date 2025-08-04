#!/bin/bash
set -e

# CUDA Toolkit 설치 (RedHat/CentOS 계열)
if ! command -v nvcc &> /dev/null; then
  echo "CUDA Toolkit이 설치되어 있지 않습니다. 설치를 시작합니다..."
  wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-rhel7-12-4-local-12.4.0_550.54.14-1.x86_64.rpm
  sudo rpm -i cuda-repo-rhel7-12-4-local-12.4.0_550.54.14-1.x86_64.rpm
  sudo yum clean all
  sudo yum -y install cuda-toolkit-12-4
fi

# Git이 설치되어 있는지 확인 후, 없으면 설치
if ! command -v git &> /dev/null; then
  echo "Git이 설치되어 있지 않습니다. 설치를 시작합니다..."
  apt-get update && apt-get install -y git
fi

# Git 인증 정보 및 저장소 URL
REPO_URL="https://sonsj97:qwaszx2689%21@gitlab.x2bee.com/tech-team/polar/polar-trainer.git"

# LLM 평가 관련 git 저장소 클론
if [ ! -d "lm-evaluation-harness" ]; then
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
  cd lm-evaluation-harness
  pip install -e .
  cd ..
else
  echo "lm-evaluation-harness 디렉터리가 존재하므로 업데이트(pull)합니다..."
  cd lm-evaluation-harness
  # 원격 저장소 URL에 인증 정보 업데이트
  #git remote set-url origin "${REPO_URL}"
  git pull
  cd ..
fi

# polar-trainer 디렉터리가 없으면 클론, 있으면 업데이트(pull)
if [ ! -d "polar-trainer" ]; then
  echo "polar-trainer 디렉터리가 없으므로 클론합니다..."
  git clone -b no_gaudi --single-branch "${REPO_URL}" polar-trainer
else
  echo "polar-trainer 디렉터리가 존재하므로 업데이트(pull)합니다..."
  cd polar-trainer
  # 원격 저장소 URL에 인증 정보 업데이트
  git remote set-url origin "${REPO_URL}"
  git pull
  cd ..
fi

# requirements.txt 설치 (가상환경 내)
echo "필수 패키지를 설치합니다..."
pip install -r polar-trainer/requirements.txt

# train_api.py 실행 (가상환경 내)
echo "train_api.py 실행합니다..."
python polar-trainer/main.py