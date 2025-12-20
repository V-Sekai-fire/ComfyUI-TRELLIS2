# AGENTS.md

This file provides AI coding agents with specific instructions and context for working with the ComfyUI-TRELLIS2 project.

## Project Overview

ComfyUI-TRELLIS2 is a ComfyUI custom node implementation for Microsoft's TRELLIS.2 image-to-3D generation model. It generates high-quality 3D meshes with PBR materials from single images.

**Key Components:**

- `install.py`: Main installation script that handles CUDA package installation with automatic wheel detection and compilation fallback
- `install_compilers.py`: Helper script to install C++ compilers for CUDA extension compilation
- `nodes/`: ComfyUI node implementations
- `trellis2/`: Core TRELLIS.2 model implementation

## Setup Commands

### Standard Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run main installation script (handles CUDA packages automatically)
python install.py
```

### Using Conda/Mamba (Recommended for Python 3.12+)

```bash
# Create environment with Python 3.12
mamba create -n comfyui python=3.12 -y
mamba activate comfyui

# Install dependencies and run installer
pip install -r requirements.txt
python install.py
```

### Manual Compiler Installation (if needed)

```bash
# Install C++ compilers for CUDA extension compilation
python install_compilers.py
```

## Build and Test Commands

### Installation Process

The `install.py` script automatically:

1. Detects PyTorch and CUDA versions
2. Attempts to install pre-built wheels for CUDA extensions
3. Falls back to compilation from source if wheels unavailable
4. Handles Fedora/Debian/Ubuntu Linux distributions
5. Installs compatible GCC versions if needed

### Testing Installation

```bash
# Verify installation
python -c "import nvdiffrast; import flex_gemm; import cumesh; print('OK')"
```

## Platform-Specific Setup

### Fedora Linux

The installation script includes Fedora support with:

- Automatic detection of Fedora version
- Fallback to older Fedora versions if repository unavailable (e.g., Fedora 43 → 42 → 40 → 39)
- DNF-based CUDA toolkit installation
- RPM Fusion repository integration
- Automatic GCC compatibility handling (installs GCC 12/13 if GCC 15+ detected)

**Fedora-specific commands:**

```bash
# Install CUDA-compatible GCC (if automatic installation fails)
sudo dnf copr enable -y kwizart/cuda-gcc-10.1
sudo dnf install -y cuda-gcc cuda-gcc-c++
```

### Debian/Ubuntu Linux

Uses `apt-get` for package management:

- Automatic CUDA repository setup
- Build tools installation via `build-essential`

### Windows

- Uses Visual Studio Build Tools
- CUDA network installer for minimal components
- See `install_compilers.py` for Windows compiler setup

## Code Style Guidelines

- **Python**: Follow PEP 8 style guide
- **File structure**:
  - Installation scripts in root directory
  - Node implementations in `nodes/`
  - Core model code in `trellis2/`
- **Error handling**: Use descriptive error messages with `[ComfyUI-TRELLIS2]` prefix
- **Logging**: Print status messages for all installation steps
- **Documentation**: Include docstrings for all functions

## Key Installation Functions

### `install.py` Main Functions

- `detect_linux_distro()`: Detects Fedora, Debian, or Ubuntu
- `try_install_cuda_toolkit()`: Installs CUDA compiler components
- `_install_cuda_fedora()`: Fedora-specific CUDA installation with version fallback
- `_install_cuda_debian()`: Debian/Ubuntu CUDA installation
- `try_install_compatible_gcc()`: Installs CUDA-compatible GCC (12/13) if GCC 15+ detected
- `try_compile_from_source()`: Compiles CUDA packages from source with proper environment setup
- `find_cuda_home()`: Locates CUDA installation (prefers CUDA 12.x over 13.x)

### Important Environment Variables

- `CUDA_HOME`: CUDA installation directory
- `CC` / `CXX`: C/C++ compiler paths
- `HOST_COMPILER`: Used when `cuda-gcc` is available
- `TORCH_CUDA_ARCH_LIST`: GPU compute capability

## Testing Instructions

### Manual Testing

1. **Test wheel installation:**

   ```bash
   python install.py
   # Check for "[OK] Installed X from wheel" messages
   ```

2. **Test compilation fallback:**

   ```bash
   # Remove a package to force compilation
   pip uninstall nvdiffrast -y
   python install.py
   # Should attempt compilation from source
   ```

3. **Test Fedora fallback:**
   ```bash
   # On Fedora 43+, should automatically fallback to Fedora 42 repository
   python install.py
   # Look for "Fedora 43 repository not available (404), trying fallback..."
   ```

### Verification

```bash
# Check installed packages
pip list | grep -E "nvdiffrast|flex_gemm|cumesh|o_voxel|flash_attn"

# Verify CUDA compiler
which nvcc
nvcc --version

# Verify GCC version
gcc --version
```

## Security Considerations

- **Sudo usage**: Installation scripts require sudo for system package installation
- **Repository verification**: CUDA repositories use GPG key verification
- **Network downloads**: All downloads are from official NVIDIA and GitHub sources
- **Temporary files**: Scripts clean up temporary files after use
- **User input**: No interactive prompts in automated installation

## Common Issues and Solutions

### GCC 15+ Incompatibility

**Problem**: CUDA doesn't support GCC 15+
**Solution**: Script automatically attempts to install GCC 12/13 or `cuda-gcc` from COPR

### Python 3.14 Compatibility

**Problem**: Many packages don't have wheels for Python 3.14
**Solution**: Use Python 3.12 or 3.13 via conda/mamba

### CUDA 12 vs 13 Mismatch

**Problem**: PyTorch uses CUDA 12.8 but system has CUDA 13.1
**Solution**: Script prefers CUDA 12.x when available, but CUDA 13.x is backward compatible for compilation

### Fedora Repository 404

**Problem**: Very new Fedora versions don't have CUDA repositories yet
**Solution**: Automatic fallback to older Fedora versions (43 → 42 → 40 → 39)

## Dependencies

### Python Packages (requirements.txt)

- huggingface_hub, hf_transfer, hf_xet
- pillow, numpy, trimesh
- imageio, imageio-ffmpeg
- opencv-python-headless
- transformers, kornia, timm
- lpips, safetensors, einops

### CUDA Packages (installed via install.py)

- nvdiffrast
- flex_gemm
- cumesh
- o_voxel
- nvdiffrec_render
- flash_attn

## File Structure

```
ComfyUI-TRELLIS2/
├── install.py              # Main installation script
├── install_compilers.py    # Compiler installation helper
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
├── nodes/                  # ComfyUI node implementations
├── trellis2/               # Core TRELLIS.2 model code
├── tests/                  # Test files
└── AGENTS.md              # This file
```

## References

- [TRELLIS.2 Repository](https://github.com/microsoft/TRELLIS.2)
- [RPM Fusion CUDA Guide](https://rpmfusion.org/Howto/CUDA)
- [agents.md Specification](https://agents.md)
