#!/usr/bin/env bash
set -e

echo "===================================="
echo "Isaac Sim + Isaac Lab Auto Installer"
echo "===================================="

WORKSPACE="/workspace/isaac"
PYTHON_VERSION="3.11"

echo ""
echo "Creating workspace directories..."
mkdir -p $WORKSPACE
mkdir -p /workspace/tmp
mkdir -p /workspace/pip_cache
mkdir -p /workspace/cache

cd $WORKSPACE

echo ""
echo "Setting persistent temp directories..."
export TMPDIR=/workspace/tmp
export TMP=/workspace/tmp
export TEMP=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip_cache

echo ""
echo "Installing system dependencies..."
apt update
apt install -y 

echo ""
echo "Installing uv..."
curl -Ls https://astral.sh/uv/install.sh | sh

source $HOME/.bashrc

echo ""
echo "Creating Python environment with uv..."
uv venv --python $PYTHON_VERSION

source .venv/bin/activate

echo ""
echo "Upgrading pip..."
uv pip install --upgrade pip

echo ""
echo "Installing Isaac Sim (this may take a while)..."
uv pip install 
--extra-index-url https://pypi.nvidia.com 
"isaacsim[all,extscache]==5.1.0"

echo ""
echo "Cloning Isaac Lab..."
git clone https://github.com/isaac-sim/IsaacLab.git

cd IsaacLab

echo ""
echo "Installing Isaac Lab Python dependencies..."
uv pip install -e .

echo ""
echo "Running Isaac Lab setup script..."
chmod +x isaaclab.sh
./isaaclab.sh --install

echo ""
echo "Configuring environment variables..."

echo 'export OMNI_KIT_ALLOW_ROOT=1' >> ~/.bashrc
echo 'export TMPDIR=/workspace/tmp' >> ~/.bashrc
echo 'export TMP=/workspace/tmp' >> ~/.bashrc
echo 'export TEMP=/workspace/tmp' >> ~/.bashrc
echo 'export PIP_CACHE_DIR=/workspace/pip_cache' >> ~/.bashrc

echo ""
echo "===================================="
echo "Installation complete!"
echo "===================================="
echo ""
echo "To activate environment in future sessions:"
echo "cd /workspace/isaac"
echo "source .venv/bin/activate"
echo ""
echo "To test Isaac Sim:"
echo "python -m isaacsim --no-window"
echo ""
echo "To test Isaac Lab:"
echo "cd IsaacLab"
echo "./isaaclab.sh --headless -p scripts/tutorials/00_sim/create_empty.py"
echo ""
