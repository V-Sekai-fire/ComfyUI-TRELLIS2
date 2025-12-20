# RunPod L40S Provisioning Guide for ComfyUI-TRELLIS2

This guide provides step-by-step instructions to provision a RunPod L40S GPU machine from scratch to run ComfyUI-TRELLIS2.

## Prerequisites

- RunPod account with access to L40S instances
- Basic familiarity with Linux command line
- SSH access to your RunPod instance

## Step 1: Initial System Setup

### Update System Packages

**For Ubuntu/Debian:**
```bash
# Update package lists
sudo apt-get update -y
sudo apt-get upgrade -y

# Install essential build tools
sudo apt-get install -y build-essential git wget curl

# Install OpenGL libraries (needed for mesh processing)
sudo apt-get install -y libgl1 libopengl0 libglu1-mesa libglx-mesa0 libosmesa6
```

**For Fedora:**
```bash
# Update package lists
sudo dnf update -y

# Install essential build tools
sudo dnf install -y gcc gcc-c++ make git wget curl

# Install OpenGL libraries (needed for mesh processing)
sudo dnf install -y mesa-libGL mesa-libGLU mesa-libGL-devel mesa-libGLU-devel
```

### Verify GPU Detection

```bash
# Check NVIDIA GPU
nvidia-smi

# Expected output should show L40S GPU with CUDA version
```

## Step 2: Install CUDA Toolkit (if not pre-installed)

### Check Existing CUDA Installation

```bash
nvcc --version
which nvcc
```

### Install CUDA 12.x (if needed)

**For Ubuntu/Debian:**
```bash
# Add NVIDIA CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA 12.8 toolkit (minimal for compilation)
sudo apt-get install -y cuda-toolkit-12-8 cuda-cudart-dev-12-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**For Fedora:**
```bash
# Add NVIDIA CUDA repository (Fedora 42+)
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/$(uname -m)/cuda-fedora42.repo
sudo dnf clean all
sudo dnf config-manager setopt cuda-fedora42-$(uname -m).exclude=nvidia-driver,nvidia-modprobe,nvidia-persistenced,nvidia-settings,nvidia-libXNVCtrl,nvidia-xconfig
sudo dnf install -y cuda-toolkit-12-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 3: Install Python 3.12 with Miniforge (Recommended)

### Download and Install Miniforge

```bash
# Download Miniforge3 (includes mamba)
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
rm Miniforge3-Linux-x86_64.sh

# Initialize conda
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### Create Python 3.12 Environment

```bash
# Create dedicated environment for ComfyUI
mamba create -n comfyui python=3.12 -y
mamba activate comfyui

# Verify Python version
python --version  # Should show Python 3.12.x
```

## Step 4: Install PyTorch with CUDA 12.8

```bash
# Make sure comfyui environment is activated
mamba activate comfyui

# Install PyTorch with CUDA 12.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch CUDA availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output:**
```
PyTorch: 2.x.x+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA L40S
```

## Step 5: Install ComfyUI

```bash
# Navigate to desired installation directory
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install ComfyUI dependencies
pip install -r requirements.txt
```

## Step 6: Install ComfyUI-TRELLIS2

```bash
# Navigate to ComfyUI custom_nodes directory
cd ~/ComfyUI/custom_nodes

# Clone ComfyUI-TRELLIS2
git clone https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2.git
cd ComfyUI-TRELLIS2

# Install dependencies and CUDA packages
python install.py
```

**Note for Fedora users:** The install script automatically detects Fedora and uses `dnf` instead of `apt-get` for CUDA toolkit installation. It will also handle GCC compatibility automatically.

**The install.py script will:**
- Install Python dependencies from `requirements.txt`
- Detect PyTorch and CUDA versions
- Attempt to install pre-built wheels for CUDA extensions
- Fall back to compilation if wheels unavailable
- Handle GCC compatibility automatically

**Expected successful output:**
```
[ComfyUI-TRELLIS2] [OK] Installed nvdiffrast from wheel
[ComfyUI-TRELLIS2] [OK] Installed flex_gemm from wheel
[ComfyUI-TRELLIS2] [OK] Installed cumesh from wheel
[ComfyUI-TRELLIS2] [OK] Installed o_voxel from wheel
[ComfyUI-TRELLIS2] [OK] Installed flash_attn from wheel
```

## Step 7: Verify Installation

```bash
# Verify all CUDA packages are installed
python -c "
import nvdiffrast
import flex_gemm
import cumesh
import o_voxel
import flash_attn
print('✓ All CUDA packages imported successfully')
"

# Check package versions
pip list | grep -E "nvdiffrast|flex_gemm|cumesh|o_voxel|flash_attn|torch"
```

## Step 8: Download TRELLIS.2 Models

The models will be automatically downloaded from HuggingFace on first use, but you can pre-download them:

```bash
# Set HuggingFace cache directory (optional)
export HF_HOME=~/ComfyUI/models/huggingface

# Models will be downloaded automatically when nodes are first used
# Or manually download using huggingface-cli if needed
```

## Step 9: Start ComfyUI

```bash
# Make sure you're in the comfyui conda environment
mamba activate comfyui

# Navigate to ComfyUI directory
cd ~/ComfyUI

# Start ComfyUI server
python main.py --listen 0.0.0.0 --port 8188
```

**For RunPod, you may want to use their template or configure port forwarding:**
- RunPod typically exposes port 8188 automatically
- Access via: `https://<your-pod-id>.runpod.net`

## Step 10: Test TRELLIS.2 Nodes

1. Open ComfyUI web interface
2. Load one of the example workflows from `ComfyUI-TRELLIS2/workflows/`:
   - `geometry_only.json` - Generate 3D geometry only
   - `geometry_texture.json` - Generate 3D with textures
   - `image_to_geometry.json` - Full image-to-3D pipeline
   - `remove_background.json` - Background removal

3. Upload a test image and run the workflow

## Troubleshooting

### Issue: CUDA packages fail to compile

**Solution for Ubuntu/Debian:**
```bash
# Install compatible GCC version
sudo apt-get install -y gcc-12 g++-12

# Re-run installation
cd ~/ComfyUI/custom_nodes/ComfyUI-TRELLIS2
python install.py
```

**Solution for Fedora:**
```bash
# Use COPR repository for CUDA-compatible GCC
sudo dnf copr enable -y kwizart/cuda-gcc-10.1
sudo dnf install -y cuda-gcc cuda-gcc-c++

# Or install GCC 12/13 if available
sudo dnf install -y gcc-12 gcc-c++-12

# Re-run installation
cd ~/ComfyUI/custom_nodes/ComfyUI-TRELLIS2
python install.py
```

### Issue: OpenGL libraries missing (for mesh processing nodes)

**Solution for Ubuntu/Debian:**
```bash
sudo apt-get install -y libgl1 libopengl0 libglu1-mesa libglx-mesa0 libosmesa6
```

**Solution for Fedora:**
```bash
sudo dnf install -y mesa-libGL mesa-libGLU mesa-libGL-devel mesa-libGLU-devel
```

### Issue: Out of Memory Errors

**Solution:**
- L40S has 48GB VRAM, but large models may still cause issues
- Reduce batch size in workflow settings
- Use model offloading if available
- Close other GPU processes: `nvidia-smi` to check

### Issue: Python version mismatch

**Solution:**
```bash
# Ensure you're using Python 3.12
mamba activate comfyui
python --version  # Should be 3.12.x

# If not, recreate environment
mamba deactivate
mamba env remove -n comfyui
mamba create -n comfyui python=3.12 -y
mamba activate comfyui
```

### Issue: Port not accessible

**Solution:**
- Check RunPod port configuration in pod settings
- Ensure ComfyUI is listening on `0.0.0.0`, not `127.0.0.1`
- Verify firewall rules in RunPod dashboard

## Quick Setup Script

### Ubuntu/Debian Setup Script

Save this as `setup_runpod_ubuntu.sh` and run it:

```bash
#!/bin/bash
set -e

echo "=== RunPod L40S ComfyUI-TRELLIS2 Setup (Ubuntu/Debian) ==="

# Update system
sudo apt-get update -y
sudo apt-get install -y build-essential git wget curl libgl1 libopengl0 libglu1-mesa libglx-mesa0 libosmesa6

# Install Miniforge
cd ~
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
rm Miniforge3-Linux-x86_64.sh
~/miniforge3/bin/conda init bash
source ~/.bashrc

# Create environment
mamba create -n comfyui python=3.12 -y
mamba activate comfyui

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install ComfyUI
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Install ComfyUI-TRELLIS2
cd custom_nodes
git clone https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2.git
cd ComfyUI-TRELLIS2
python install.py

echo "=== Setup Complete ==="
echo "To start ComfyUI:"
echo "  mamba activate comfyui"
echo "  cd ~/ComfyUI"
echo "  python main.py --listen 0.0.0.0 --port 8188"
```

### Fedora Setup Script

Save this as `setup_runpod_fedora.sh` and run it:

```bash
#!/bin/bash
set -e

echo "=== RunPod L40S ComfyUI-TRELLIS2 Setup (Fedora) ==="

# Update system
sudo dnf update -y
sudo dnf install -y gcc gcc-c++ make git wget curl mesa-libGL mesa-libGLU mesa-libGL-devel mesa-libGLU-devel

# Install Miniforge
cd ~
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
rm Miniforge3-Linux-x86_64.sh
~/miniforge3/bin/conda init bash
source ~/.bashrc

# Create environment
mamba create -n comfyui python=3.12 -y
mamba activate comfyui

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install ComfyUI
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Install ComfyUI-TRELLIS2
cd custom_nodes
git clone https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2.git
cd ComfyUI-TRELLIS2
python install.py

echo "=== Setup Complete ==="
echo "To start ComfyUI:"
echo "  mamba activate comfyui"
echo "  cd ~/ComfyUI"
echo "  python main.py --listen 0.0.0.0 --port 8188"
```

**Make executable and run:**
```bash
chmod +x setup_runpod_fedora.sh
./setup_runpod_fedora.sh
```

**Make executable and run:**
```bash
chmod +x setup_runpod_ubuntu.sh
./setup_runpod_ubuntu.sh
```

**For Fedora:**
```bash
chmod +x setup_runpod_fedora.sh
./setup_runpod_fedora.sh
```

## System Requirements Summary

- **GPU**: NVIDIA L40S (48GB VRAM)
- **CUDA**: 12.8 (compatible with PyTorch)
- **Python**: 3.12 (recommended) or 3.13
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for models and dependencies
- **OS**: Ubuntu 22.04+ or Fedora 39+ (tested)

## Estimated Setup Time

- System updates: 5-10 minutes
- CUDA installation: 10-15 minutes (if needed)
- Miniforge + Python 3.12: 5 minutes
- PyTorch installation: 5-10 minutes
- ComfyUI installation: 5-10 minutes
- ComfyUI-TRELLIS2 installation: 10-20 minutes (wheels) or 30-60 minutes (compilation)

**Total: ~45-90 minutes depending on compilation needs**

## Next Steps

1. Configure RunPod persistent storage for models
2. Set up automatic model downloads
3. Create custom workflows for your use case
4. Monitor GPU usage with `nvidia-smi`
5. Set up model caching to reduce download time

## References

- [ComfyUI Repository](https://github.com/comfyanonymous/ComfyUI)
- [TRELLIS.2 Repository](https://github.com/microsoft/TRELLIS.2)
- [RunPod Documentation](https://docs.runpod.io)
- [AGENTS.md](./AGENTS.md) - For AI agent development guidance

