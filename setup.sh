#!/bin/bash

echo "Starting setup script..."
set -e  # Exit on any error

if command -v module >/dev/null 2>&1; then
  echo "Loading python 3.12.3..."
  module load devel/python/3.12.3-gnu-14.2
else
  echo "Module util is not available. Installing python 3.12.3..."
  wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz || {
    echo "[ERROR] Failed to download Python source." >&2
    exit 1
}
  tar -xzf Python-3.12.3.tgz
  cd Python-3.12.3 || {
      echo "[ERROR] Failed to enter Python source directory." >&2
      exit 1
  }
  ./configure --enable-optimizations --prefix="$HOME/.local/python3.12" || {
    echo "[ERROR] ./configure failed." >&2
    exit 1
  }
  make -j"$(nproc)" || {
    echo "[ERROR] make failed." >&2
    exit 1
  }
  make install || {
    echo "[ERROR] make install failed." >&2
    exit 1
  }
  export PATH="$HOME/.local/python3.12/bin:$PATH"
  if ! grep -q 'python3.12/bin' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/python3.12/bin:$PATH"' >> ~/.bashrc
  fi
  source ~/.bashrc 2>/dev/null

  cd ..
  rm -rf Python-3.12.3 Python-3.12.3.tgz
  echo "Python installed successfully: $(python3.12 --version)"
fi

if pwd | grep -q "research-project"; then
  echo "You are in the research-project directory."
else
  echo "Navigating to the research-project directory."
  cd ~/research-project || exit 1
fi

ENV_NAME=".env"
echo "Creating virtual environment '$ENV_NAME'..."
python3.12 -m venv $ENV_NAME
source $ENV_NAME/bin/activate

echo "Installing pip..."
python3.12 -m ensurepip --upgrade
pip3.12 install --upgrade pip

DATA_DIR="$HOME/tasks_1-20_v1-2"
echo "Checking for data directory: $DATA_DIR"
if [ -d "$DATA_DIR" ]; then
    echo "Data directory '$DATA_DIR' exists. Skipping download."
else
  echo "Downloading bAbI tasks..."
  pip3.12 install kagglehub
  python3.12 -c "import kagglehub; kagglehub.dataset_download(\"roblexnana/the-babi-tasks-for-nlp-qa-system\")"
  BABI_DIR=~/.cache/kagglehub/datasets/roblexnana/the-babi-tasks-for-nlp-qa-system/versions/1
  mv $BABI_DIR/tasks_1-20_v1-2 ~/tasks_1-20_v1-2
fi
echo "The bAbI tasks are in ~/tasks_1-20_v1-2"

if command -v module >/dev/null 2>&1; then
  echo "Loading CUDA 12.8..."
  module load devel/cuda/12.8
else
  echo "Module util is not available. Find how to install CUDA 12.8 manually."
fi

echo "Installing torch libraries..."
pip3.12 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 || {
    echo "[ERROR] Failed to install a torch library." >&2
    exit 1
}

python3.12 -m pip install -U pip setuptools wheel || {
    echo "[ERROR] Failed to install pip, setuptools, or wheel." >&2
    exit 1
}
echo "Installing spaCy..."
pip3.12 install -U spacy || {
    echo "[ERROR] Failed to install spaCy." >&2
    exit 1
}
echo "Installing CuPy with CUDA 12.x support..."
pip3.12 install cupy-cuda12x || {
    echo "[ERROR] Failed to install CuPy for CUDA 12.x." >&2
    exit 1
}
echo "Downloading spaCy language model (en_core_web_sm)..."
python3.12 -m spacy download en_core_web_sm || {
    echo "[ERROR] Failed to download spaCy model 'en_core_web_sm'." >&2
    exit 1
}

echo "Installing other requirements from requirements.txt..."
pip3.12 install -r requirements.txt || {
    echo "[ERROR] Failed to install requirements from requirements.txt." >&2
    exit 1
}

echo "Updating code formatting tools (black, flake8, isort)..."
pip3.12 install -U black flake8 isort || {
    echo "[ERROR] Failed to update formatting tools." >&2
    exit 1
}

echo "Let's install nltk corpora. Go for 'd' option (download) and type 'all'."
python3.12 -m nltk.downloader

read -rp "Enter your Huggingface token: " token
export HUGGINGFACE="$token"

huggingface-cli login --token "$HUGGINGFACE"

echo "✅ All packages installed successfully!"
echo "The setup is finished. Good luck on your research journey, adventurous traveller!"