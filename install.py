#!/usr/bin/env python3
"""
Installation script for ComfyUI-TRELLIS2.
Called by ComfyUI Manager during installation/update.

Automatically detects PyTorch/CUDA versions and installs pre-built wheels
for CUDA extensions (nvdiffrast, flex_gemm, cumesh, o_voxel, nvdiffrec_render).
Falls back to compilation from source if no wheel is available.
"""
import subprocess
import sys
import os
import site
import shutil
import re
import tempfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path


# =============================================================================
# Package Configuration
# =============================================================================

# GitHub release URL template: {repo}/releases/download/{tag}/{wheel_name}
# wheel_name format: {package}-{version}+{cuda_suffix}-cp{py}-cp{py}-{platform}.whl
PACKAGES = [
    {
        "name": "nvdiffrast",
        "import_name": "nvdiffrast",
        "wheel_index": "https://pozzettiandrea.github.io/nvdiffrast-full-wheels/",
        "wheel_release_base": "https://github.com/PozzettiAndrea/nvdiffrast-full-wheels/releases/download",
        "wheel_version": "0.4.0",
        "git_url": "git+https://github.com/NVlabs/nvdiffrast",
    },
    {
        "name": "flex_gemm",
        "import_name": "flex_gemm",
        "wheel_index": "https://pozzettiandrea.github.io/flexgemm-wheels/",
        "wheel_release_base": "https://github.com/PozzettiAndrea/flexgemm-wheels/releases/download",
        "wheel_version": "0.0.1",
        "git_url": "git+https://github.com/JeffreyXiang/FlexGEMM",
    },
    {
        "name": "cumesh",
        "import_name": "cumesh",
        "wheel_index": "https://pozzettiandrea.github.io/cumesh-wheels/",
        "wheel_release_base": "https://github.com/PozzettiAndrea/cumesh-wheels/releases/download",
        "wheel_version": "0.0.1",
        "git_url": "git+https://github.com/JeffreyXiang/CuMesh",
    },
    {
        "name": "o_voxel",
        "import_name": "o_voxel",
        "wheel_index": "https://pozzettiandrea.github.io/ovoxel-wheels/",
        "wheel_release_base": "https://github.com/PozzettiAndrea/ovoxel-wheels/releases/download",
        "wheel_version": "0.0.1",
        "git_url": "git+https://github.com/microsoft/TRELLIS.2#subdirectory=o-voxel",
    },
    {
        "name": "nvdiffrec_render",
        "import_name": "nvdiffrec_render",
        "wheel_index": "https://pozzettiandrea.github.io/nvdiffrec_render-wheels/",
        "wheel_release_base": "https://github.com/PozzettiAndrea/nvdiffrec_render-wheels/releases/download",
        "wheel_version": "0.0.1",
        "git_url": None,  # Source requires custom build process, wheel-only
    },
    {
        "name": "flash_attn",
        "import_name": "flash_attn",
        "wheel_release_base": "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download",
        "wheel_type": "flash_attn",  # Special handling for torch version in wheel name
        "git_url": "git+https://github.com/Dao-AILab/flash-attention",
    },
]

# Map (CUDA major.minor, PyTorch major.minor) -> wheel CUDA suffix
# Wheels are built for cu124 and cu128
WHEEL_CUDA_MAP = {
    ("12.4", "2.5"): "cu124",
    ("12.6", "2.6"): "cu124",
    ("12.6", "2.8"): "cu128",
    ("12.8", "2.8"): "cu128",
    ("12.8", "2.9"): "cu128",
    ("12.9", "2.8"): "cu128",  # CUDA 12.9 can use 12.8 wheels
    ("12.9", "2.9"): "cu128",
}

# Map (CUDA major.minor, PyTorch major.minor) -> wheel subdirectory
# Static mapping ensures any patch version (2.9.0, 2.9.1, etc.) finds the right wheels
WHEEL_DIRS = {
    ("12.4", "2.5"): "cu124-torch251",
    ("12.6", "2.6"): "cu126-torch260",
    ("12.6", "2.8"): "cu126-torch280",
    ("12.8", "2.8"): "cu128-torch280",
    ("12.8", "2.9"): "cu128-torch291",
    ("12.9", "2.8"): "cu128-torch280",  # CUDA 12.9 uses 12.8 wheels
    ("12.9", "2.9"): "cu128-torch291",
}

# Flash attention wheel sources - tried in order until one works
# Different repos use different naming conventions
FLASH_ATTN_SOURCES = [
    {
        "name": "bdashore3",
        "base_url": "https://github.com/bdashore3/flash-attention/releases/download",
        "versions": [("2.8.3", "v2.8.3")],
        # Format: flash_attn-{ver}+cu{cuda}torch{torch}.0cxx11abiFALSE-cp{py}-cp{py}-{platform}.whl
        "format": "cxx11abi",
        "platforms": ["win_amd64", "linux_x86_64"],
    },
    {
        "name": "mjun0812",
        "base_url": "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download",
        "versions": [("2.8.3", "v0.4.19"), ("2.7.4.post1", "v0.4.19")],
        # Format: flash_attn-{ver}+cu{cuda}torch{torch}-cp{py}-cp{py}-{platform}.whl
        "format": "simple",
        "platforms": ["win_amd64", "linux_x86_64"],
    },
    {
        "name": "oobabooga",
        "base_url": "https://github.com/oobabooga/flash-attention/releases/download",
        "versions": [("2.7.4.post1", "v2.7.4.post1"), ("2.7.3", "v2.7.3")],
        # Format: flash_attn-{ver}+cu{cuda}torch{torch}.0cxx11abiFALSE-cp{py}-cp{py}-{platform}.whl
        "format": "cxx11abi",
        "platforms": ["win_amd64", "linux_x86_64"],
    },
]


# =============================================================================
# Version Detection
# =============================================================================

def get_python_version():
    """Get Python version as major.minor string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def detect_linux_distro():
    """
    Detect Linux distribution.
    Returns 'fedora', 'debian', 'ubuntu', or None.
    """
    if sys.platform != "linux":
        return None
    
    # Check /etc/os-release (most reliable)
    try:
        with open("/etc/os-release", "r") as f:
            content = f.read()
            if "fedora" in content.lower():
                return "fedora"
            elif "debian" in content.lower():
                return "debian"
            elif "ubuntu" in content.lower():
                return "ubuntu"
    except (FileNotFoundError, IOError):
        pass
    
    # Fallback: check for distribution-specific files
    if os.path.exists("/etc/fedora-release"):
        return "fedora"
    elif os.path.exists("/etc/debian_version"):
        return "debian"
    elif os.path.exists("/etc/lsb-release"):
        try:
            with open("/etc/lsb-release", "r") as f:
                if "ubuntu" in f.read().lower():
                    return "ubuntu"
        except (FileNotFoundError, IOError):
            pass
    
    return None


def get_torch_info():
    """
    Get PyTorch and CUDA version info.
    Returns (torch_version, cuda_version) or (None, None) if not available.
    """
    try:
        import torch
        torch_ver = torch.__version__.split('+')[0]  # Remove +cu128 suffix if present
        cuda_ver = torch.version.cuda
        return torch_ver, cuda_ver
    except ImportError:
        return None, None


def get_wheel_cuda_suffix():
    """
    Determine which CUDA wheel suffix to use (cu124 or cu128).
    Returns suffix string or None if no matching wheel available.
    """
    torch_ver, cuda_ver = get_torch_info()
    if not torch_ver or not cuda_ver:
        return None

    # Extract major.minor versions
    cuda_mm = '.'.join(cuda_ver.split('.')[:2])
    torch_mm = '.'.join(torch_ver.split('.')[:2])

    suffix = WHEEL_CUDA_MAP.get((cuda_mm, torch_mm))
    if suffix:
        print(f"[ComfyUI-TRELLIS2] Detected CUDA {cuda_mm} + PyTorch {torch_mm} -> using {suffix} wheels")
        return suffix

    # Try to find closest match by CUDA version alone
    if cuda_mm.startswith("12.4"):
        print(f"[ComfyUI-TRELLIS2] CUDA {cuda_mm} detected, trying cu124 wheels")
        return "cu124"
    elif cuda_mm.startswith("12.") and float(cuda_mm.split('.')[1]) >= 6:
        print(f"[ComfyUI-TRELLIS2] CUDA {cuda_mm} detected, trying cu128 wheels")
        return "cu128"

    print(f"[ComfyUI-TRELLIS2] No matching wheel for CUDA {cuda_mm} + PyTorch {torch_mm}")
    return None


def get_wheel_dir():
    """
    Get the wheel subdirectory for current CUDA + PyTorch version.
    Uses static mapping so any patch version (2.9.0, 2.9.1) maps to the same wheels.
    Returns directory string (e.g., 'cu128-torch291') or None if no match.
    """
    torch_ver, cuda_ver = get_torch_info()
    if not torch_ver or not cuda_ver:
        return None

    cuda_mm = '.'.join(cuda_ver.split('.')[:2])
    torch_mm = '.'.join(torch_ver.split('.')[:2])

    wheel_dir = WHEEL_DIRS.get((cuda_mm, torch_mm))
    if wheel_dir:
        print(f"[ComfyUI-TRELLIS2] Using wheel directory: {wheel_dir}")
        return wheel_dir

    return None


# =============================================================================
# CUDA Environment (for compilation fallback)
# =============================================================================

def find_cuda_home():
    """Find CUDA installation directory."""
    # Check CUDA_HOME environment variable
    if "CUDA_HOME" in os.environ and os.path.exists(os.environ["CUDA_HOME"]):
        cuda_home = os.environ["CUDA_HOME"]
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(nvcc_path) or os.path.exists(nvcc_path + ".exe"):
            return cuda_home

    # Check nvcc in PATH
    nvcc_in_path = shutil.which("nvcc")
    if nvcc_in_path:
        nvcc_dir = os.path.dirname(nvcc_in_path)
        if os.path.basename(nvcc_dir) == "bin":
            cuda_home = os.path.dirname(nvcc_dir)
            if cuda_home != "/usr":
                return cuda_home

    # Check system CUDA installations
    # First, check for any cuda-* directories and use the newest one
    cuda_base = "/usr/local"
    if os.path.exists(cuda_base):
        cuda_dirs = []
        for item in os.listdir(cuda_base):
            if item.startswith("cuda-") and os.path.isdir(os.path.join(cuda_base, item)):
                cuda_path = os.path.join(cuda_base, item)
                nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
                if os.path.exists(nvcc_path):
                    cuda_dirs.append(cuda_path)
        # Sort by version (newest first) - extract version number
        def get_version(path):
            version_str = os.path.basename(path).replace("cuda-", "")
            try:
                # Try to parse as float (e.g., "13.1" -> 13.1)
                parts = version_str.split('.')
                if len(parts) >= 2:
                    return float(f"{parts[0]}.{parts[1]}")
                elif len(parts) == 1:
                    return float(parts[0])
            except:
                return 0.0
            return 0.0
        
        # Prefer CUDA 12.x over 13.x for compatibility
        # Separate CUDA 12.x and 13.x versions
        cuda_12_dirs = [d for d in cuda_dirs if get_version(d) >= 12.0 and get_version(d) < 13.0]
        cuda_13_dirs = [d for d in cuda_dirs if get_version(d) >= 13.0]
        other_dirs = [d for d in cuda_dirs if get_version(d) < 12.0]
        
        # Sort each group (newest first)
        cuda_12_dirs.sort(key=get_version, reverse=True)
        cuda_13_dirs.sort(key=get_version, reverse=True)
        other_dirs.sort(key=get_version, reverse=True)
        
        # Return CUDA 12.x first, then 13.x, then others
        if cuda_12_dirs:
            print(f"[ComfyUI-TRELLIS2] Using CUDA 12.x: {cuda_12_dirs[0]}")
            return cuda_12_dirs[0]
        elif cuda_13_dirs:
            print(f"[ComfyUI-TRELLIS2] Using CUDA 13.x: {cuda_13_dirs[0]} (CUDA 12.x not found)")
            return cuda_13_dirs[0]
        elif other_dirs:
            return other_dirs[0]
    
    # Fallback to specific versions (prefer CUDA 12.x)
    system_paths = [
        "/usr/local/cuda-12.8", "/usr/local/cuda-12.6", "/usr/local/cuda-12.4", 
        "/usr/local/cuda-12.2", "/usr/local/cuda-12.0", "/usr/local/cuda-13.1", 
        "/usr/local/cuda-13.0", "/usr/local/cuda",
    ]
    for cuda_path in system_paths:
        if os.path.exists(cuda_path):
            nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
            if os.path.exists(nvcc_path):
                return cuda_path

    # Check Windows locations
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        cuda_base = os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
        if os.path.exists(cuda_base):
            versions = sorted(os.listdir(cuda_base), reverse=True)
            for v in versions:
                cuda_path = os.path.join(cuda_base, v)
                if os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
                    return cuda_path

    # Check pip-installed CUDA
    for sp in site.getsitepackages():
        for subdir in ["nvidia/cuda_nvcc", "nvidia_cuda_nvcc"]:
            cuda_path = os.path.join(sp, subdir)
            if os.path.exists(cuda_path):
                nvcc = os.path.join(cuda_path, "bin", "nvcc")
                if os.path.exists(nvcc) or os.path.exists(nvcc + ".exe"):
                    return cuda_path

    # Check conda
    if "CONDA_PREFIX" in os.environ:
        nvcc = os.path.join(os.environ["CONDA_PREFIX"], "bin", "nvcc")
        if os.path.exists(nvcc):
            return os.environ["CONDA_PREFIX"]

    return None


def try_install_cuda_toolkit():
    """
    Try to install minimal CUDA toolkit components needed for compilation.
    - Linux: Uses apt (Debian/Ubuntu) or dnf (Fedora) to install CUDA components
    - Windows: Downloads and runs CUDA network installer with minimal components
    Returns True if successful, False otherwise.
    """
    print("[ComfyUI-TRELLIS2] CUDA compiler not found, attempting to install...")

    # Get CUDA version from PyTorch
    try:
        import torch
        cuda_ver = torch.version.cuda
        if not cuda_ver:
            print("[ComfyUI-TRELLIS2] PyTorch CUDA version not available")
            return False
        cuda_parts = cuda_ver.split('.')
        cuda_major = cuda_parts[0]  # "12"
        cuda_minor = cuda_parts[1] if len(cuda_parts) > 1 else "0"  # "8"
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Could not detect CUDA version: {e}")
        return False

    if sys.platform == "linux":
        distro = detect_linux_distro()
        return _install_cuda_linux(cuda_major, cuda_minor, distro)
    elif sys.platform == "win32":
        return _install_cuda_windows(cuda_major, cuda_minor)
    else:
        print(f"[ComfyUI-TRELLIS2] Unsupported platform: {sys.platform}")
        return False


def _install_cuda_linux(cuda_major, cuda_minor, distro=None):
    """
    Install minimal CUDA components on Linux.
    Supports Debian/Ubuntu (apt) and Fedora (dnf).
    """
    if distro == "fedora":
        return _install_cuda_fedora(cuda_major, cuda_minor)
    else:
        # Default to Debian/Ubuntu (apt)
        return _install_cuda_debian(cuda_major, cuda_minor)


def _install_cuda_debian(cuda_major, cuda_minor):
    """Install minimal CUDA components on Debian/Ubuntu via apt."""
    print("[ComfyUI-TRELLIS2] Installing CUDA compiler via apt (minimal components)...")

    # Add NVIDIA repository if not present
    keyring_path = "/usr/share/keyrings/cuda-archive-keyring.gpg"
    if not os.path.exists(keyring_path):
        print("[ComfyUI-TRELLIS2] Adding NVIDIA CUDA repository...")
        try:
            # Download and install keyring
            subprocess.run([
                "wget", "-q", "-O", "/tmp/cuda-keyring.deb",
                "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
            ], check=True, timeout=60)
            subprocess.run(["sudo", "dpkg", "-i", "/tmp/cuda-keyring.deb"], check=True, timeout=30)
            subprocess.run(["sudo", "apt-get", "update", "-qq"], check=True, timeout=120)
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI-TRELLIS2] Failed to add NVIDIA repository: {e}")
            return False
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] Repository setup error: {e}")
            return False

    # Install minimal CUDA components
    packages = [
        f"cuda-nvcc-{cuda_major}-{cuda_minor}",
        f"cuda-cudart-dev-{cuda_major}-{cuda_minor}",
    ]

    print(f"[ComfyUI-TRELLIS2] Installing: {', '.join(packages)}")
    try:
        result = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "-qq"] + packages,
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"[ComfyUI-TRELLIS2] apt install failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
            return False
    except subprocess.TimeoutExpired:
        print("[ComfyUI-TRELLIS2] Installation timed out")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Installation error: {e}")
        return False

    # Update PATH
    cuda_path = f"/usr/local/cuda-{cuda_major}.{cuda_minor}"
    if os.path.exists(cuda_path):
        os.environ["CUDA_HOME"] = cuda_path
        os.environ["PATH"] = f"{cuda_path}/bin:" + os.environ.get("PATH", "")

    print("[ComfyUI-TRELLIS2] [OK] CUDA compiler installed via apt")
    return True


def _install_cuda_fedora(cuda_major, cuda_minor):
    """
    Install minimal CUDA components on Fedora via dnf.
    Follows RPM Fusion recommendations: https://rpmfusion.org/Howto/CUDA
    """
    print("[ComfyUI-TRELLIS2] Installing CUDA compiler via dnf (minimal components)...")

    # Get Fedora version for repository URL
    fedora_version = None
    try:
        with open("/etc/os-release", "r") as f:
            for line in f:
                if line.startswith("VERSION_ID="):
                    fedora_version = line.split("=")[1].strip().strip('"')
                    break
    except (FileNotFoundError, IOError):
        pass

    if not fedora_version:
        # Try to detect from /etc/fedora-release
        try:
            with open("/etc/fedora-release", "r") as f:
                content = f.read()
                # Extract version number (e.g., "Fedora release 43 (Forty Three)" -> "43")
                match = re.search(r'\d+', content)
                if match:
                    fedora_version = match.group()
        except (FileNotFoundError, IOError):
            pass

    if not fedora_version:
        print("[ComfyUI-TRELLIS2] Could not detect Fedora version, using default repository")
        fedora_version = "40"  # Default fallback

    try:
        fedora_version_int = int(fedora_version)
    except (ValueError, TypeError):
        fedora_version_int = 40

    # Try Fedora versions in order: current version, then fallbacks
    # NVIDIA may not have repositories for very new Fedora versions
    fedora_versions_to_try = [fedora_version_int]
    if fedora_version_int >= 43:
        fedora_versions_to_try.extend([42, 40, 39])  # Try older versions as fallback
    elif fedora_version_int == 42:
        fedora_versions_to_try.extend([40, 39])
    elif fedora_version_int >= 40:
        fedora_versions_to_try.extend([39])
    
    # Check if repository is already configured
    repo_exists = False
    repo_version_used = None
    try:
        result = subprocess.run(
            ["dnf", "repolist", "--enabled"],
            capture_output=True, text=True, timeout=30
        )
        for fv in fedora_versions_to_try:
            repo_name = f"cuda-fedora{fv}"
            if repo_name in result.stdout or f"cuda-fedora{fv}-x86_64" in result.stdout:
                repo_exists = True
                repo_version_used = fv
                break
    except:
        pass

    if not repo_exists:
        print("[ComfyUI-TRELLIS2] Adding NVIDIA CUDA repository...")
        # Try each Fedora version until one works
        repo_added = False
        for fv in fedora_versions_to_try:
            repo_url = f"https://developer.download.nvidia.com/compute/cuda/repos/fedora{fv}/x86_64/cuda-fedora{fv}.repo"
            try:
                # First, check if the URL exists by trying to fetch it
                try:
                    req = urllib.request.Request(repo_url)
                    req.add_header('User-Agent', 'Mozilla/5.0')
                    with urllib.request.urlopen(req, timeout=10) as response:
                        if response.status == 200:
                            # URL exists, proceed with adding repo
                            pass
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        print(f"[ComfyUI-TRELLIS2] Fedora {fv} repository not available (404), trying fallback...")
                        continue
                    else:
                        raise
                except Exception:
                    # If we can't check, try anyway
                    pass
                
                # Use dnf config-manager to add repository (official method)
                # For Fedora 42+ with DNF-5, use --from-repofile
                if fv >= 42:
                    # Fedora 42+ method
                    result = subprocess.run(
                        ["sudo", "dnf", "config-manager", "addrepo", "--from-repofile", repo_url],
                        capture_output=True, text=True, timeout=60
                    )
                    if result.returncode == 0:
                        subprocess.run(
                            ["sudo", "dnf", "clean", "all"],
                            check=True, timeout=30
                        )
                        # Exclude nvidia driver packages (use RPM Fusion driver instead)
                        arch = subprocess.run(["uname", "-m"], capture_output=True, text=True).stdout.strip()
                        exclude_repo = f"cuda-fedora{fv}-{arch}"
                        subprocess.run(
                            ["sudo", "dnf", "config-manager", "setopt", 
                             f"{exclude_repo}.exclude=nvidia-driver,nvidia-modprobe,nvidia-persistenced,nvidia-settings,nvidia-libXNVCtrl,nvidia-xconfig"],
                            check=True, timeout=30
                        )
                        repo_added = True
                        repo_version_used = fv
                        print(f"[ComfyUI-TRELLIS2] Successfully added CUDA repository for Fedora {fv}")
                        break
                    else:
                        if "404" in result.stderr or "not found" in result.stderr.lower():
                            print(f"[ComfyUI-TRELLIS2] Fedora {fv} repository not available, trying fallback...")
                            continue
                        else:
                            raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)
                else:
                    # Fedora 39+ method
                    result = subprocess.run(
                        ["sudo", "dnf", "config-manager", "--add-repo", repo_url],
                        capture_output=True, text=True, timeout=60
                    )
                    if result.returncode == 0:
                        subprocess.run(
                            ["sudo", "dnf", "clean", "all"],
                            check=True, timeout=30
                        )
                        # Disable nvidia-driver module (use RPM Fusion driver instead)
                        subprocess.run(
                            ["sudo", "dnf", "module", "disable", "nvidia-driver", "-y"],
                            check=True, timeout=30
                        )
                        repo_added = True
                        repo_version_used = fv
                        print(f"[ComfyUI-TRELLIS2] Successfully added CUDA repository for Fedora {fv}")
                        break
                    else:
                        if "404" in result.stderr or "not found" in result.stderr.lower():
                            print(f"[ComfyUI-TRELLIS2] Fedora {fv} repository not available, trying fallback...")
                            continue
                        else:
                            raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)
            except subprocess.CalledProcessError as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    print(f"[ComfyUI-TRELLIS2] Fedora {fv} repository not available, trying fallback...")
                    continue
                else:
                    # If it's not a 404, try the next version
                    print(f"[ComfyUI-TRELLIS2] Failed to add Fedora {fv} repository: {e}")
                    continue
            except Exception as e:
                print(f"[ComfyUI-TRELLIS2] Error checking Fedora {fv} repository: {e}")
                continue
        
        if not repo_added:
            print(f"[ComfyUI-TRELLIS2] Failed to add NVIDIA repository for any Fedora version tried: {fedora_versions_to_try}")
            print("[ComfyUI-TRELLIS2] You may need to manually add the CUDA repository or wait for NVIDIA to support your Fedora version")
            return False

    # Install minimal CUDA components
    # Use the version that was successfully configured (or current if already existed)
    if repo_version_used is None:
        repo_version_used = fedora_version_int
    
    # Prefer CUDA 12.x over 13.x for compatibility with PyTorch CUDA 12.8
    # Try to install CUDA 12.8 first (matching PyTorch), then fallback to other 12.x versions
    cuda_12_versions = ["12.8", "12.6", "12.4", "12.2", "12.0"]
    
    # For Fedora 42+, use cuda-toolkit; for older versions, use specific packages
    if repo_version_used >= 42:
        # Fedora 42+ uses cuda-toolkit package
        # Try CUDA 12.x toolkit first (prefer 12.8 to match PyTorch)
        packages = []
        for cuda_12_ver in cuda_12_versions:
            major, minor = cuda_12_ver.split('.')
            # Try different package name formats
            pkg_names = [
                f"cuda-toolkit-{major}-{minor}",  # e.g., cuda-toolkit-12-8
                f"cuda-{major}-{minor}",          # e.g., cuda-12-8
                f"cuda-toolkit-{major}.{minor}",  # e.g., cuda-toolkit-12.8
            ]
            for pkg_name in pkg_names:
                result = subprocess.run(
                    ["dnf", "list", "available", pkg_name],
                    capture_output=True, text=True, timeout=30
                )
                if pkg_name in result.stdout:
                    packages = [pkg_name]
                    print(f"[ComfyUI-TRELLIS2] Installing CUDA 12.{minor} toolkit ({pkg_name})")
                    break
            if packages:
                break
        
        # If CUDA 12.x not available, try installing individual CUDA 12.x components
        if not packages:
            print("[ComfyUI-TRELLIS2] CUDA 12.x toolkit packages not found, trying individual components...")
            for cuda_12_ver in cuda_12_versions:
                major, minor = cuda_12_ver.split('.')
                pkg_nvcc = f"cuda-nvcc-{major}-{minor}"
                pkg_cudart = f"cuda-cudart-devel-{major}-{minor}"
                result = subprocess.run(
                    ["dnf", "list", "available", pkg_nvcc, pkg_cudart],
                    capture_output=True, text=True, timeout=30
                )
                if pkg_nvcc in result.stdout and pkg_cudart in result.stdout:
                    packages = [pkg_nvcc, pkg_cudart]
                    print(f"[ComfyUI-TRELLIS2] Installing CUDA 12.{minor} components (nvcc + cudart-devel)")
                    break
        
        # Fallback to latest if CUDA 12 not available
        if not packages:
            packages = ["cuda-toolkit"]
            print("[ComfyUI-TRELLIS2] [WARNING] CUDA 12.x not available in repository")
            print("[ComfyUI-TRELLIS2] [WARNING] Installing latest CUDA toolkit (may be CUDA 13.x)")
            print("[ComfyUI-TRELLIS2] [WARNING] This may cause compatibility issues with PyTorch CUDA 12.8")
    else:
        # Fedora 39+: Use specific CUDA packages
        # Try CUDA 12.x first (prefer 12.8 to match PyTorch)
        packages = []
        for cuda_12_ver in cuda_12_versions:
            major, minor = cuda_12_ver.split('.')
            pkg_nvcc = f"cuda-nvcc-{major}-{minor}"
            pkg_cudart = f"cuda-cudart-devel-{major}-{minor}"
            # Check if packages exist
            result = subprocess.run(
                ["dnf", "list", "available", pkg_nvcc, pkg_cudart],
                capture_output=True, text=True, timeout=30
            )
            if pkg_nvcc in result.stdout and pkg_cudart in result.stdout:
                packages = [pkg_nvcc, pkg_cudart]
                print(f"[ComfyUI-TRELLIS2] Installing CUDA 12.{minor} components")
                break
        
        # Fallback to requested version if CUDA 12 not available
        if not packages:
            packages = [
                f"cuda-nvcc-{cuda_major}-{cuda_minor}",
                f"cuda-cudart-devel-{cuda_major}-{cuda_minor}",
            ]
            print(f"[ComfyUI-TRELLIS2] [WARNING] CUDA 12.x not available, installing CUDA {cuda_major}.{cuda_minor}")
            print(f"[ComfyUI-TRELLIS2] [WARNING] This may cause compatibility issues with PyTorch CUDA 12.8")

    print(f"[ComfyUI-TRELLIS2] Installing: {', '.join(packages)}")
    try:
        result = subprocess.run(
            ["sudo", "dnf", "install", "-y", "-q"] + packages,
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            error_msg = result.stderr[:200] if result.stderr else result.stdout[:200] if result.stdout else "unknown error"
            print(f"[ComfyUI-TRELLIS2] dnf install failed: {error_msg}")
            return False
    except subprocess.TimeoutExpired:
        print("[ComfyUI-TRELLIS2] Installation timed out")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Installation error: {e}")
        return False

    # Update PATH
    cuda_path = f"/usr/local/cuda-{cuda_major}.{cuda_minor}"
    if os.path.exists(cuda_path):
        os.environ["CUDA_HOME"] = cuda_path
        os.environ["PATH"] = f"{cuda_path}/bin:" + os.environ.get("PATH", "")

    print("[ComfyUI-TRELLIS2] [OK] CUDA compiler installed via dnf")
    return True


def _install_cuda_windows(cuda_major, cuda_minor):
    """Install minimal CUDA components on Windows via network installer."""
    import tempfile
    import urllib.request

    print("[ComfyUI-TRELLIS2] Installing CUDA compiler via network installer (minimal components)...")

    # CUDA network installer URL (adjust version as needed)
    cuda_version = f"{cuda_major}.{cuda_minor}"
    # Map to installer version (e.g., 12.8 -> 12.8.1)
    installer_versions = {
        "12.8": "12.8.1",
        "12.6": "12.6.3",
        "12.4": "12.4.1",
    }
    installer_ver = installer_versions.get(cuda_version, f"{cuda_version}.0")

    installer_url = f"https://developer.download.nvidia.com/compute/cuda/{installer_ver}/network_installers/cuda_{installer_ver}_windows_network.exe"
    installer_path = os.path.join(tempfile.gettempdir(), "cuda_network_installer.exe")

    # Download installer
    print(f"[ComfyUI-TRELLIS2] Downloading CUDA {cuda_version} network installer...")
    try:
        urllib.request.urlretrieve(installer_url, installer_path)
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Failed to download installer: {e}")
        return False

    # Run installer with minimal components (silent mode)
    # Components: nvcc (compiler), cudart (runtime)
    components = [
        f"nvcc_{cuda_major}.{cuda_minor}",
        f"cudart_{cuda_major}.{cuda_minor}",
    ]

    print(f"[ComfyUI-TRELLIS2] Installing components: {', '.join(components)}")
    try:
        cmd = [installer_path, "-s"] + components
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"[ComfyUI-TRELLIS2] Installer failed with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("[ComfyUI-TRELLIS2] Installation timed out")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Installation error: {e}")
        return False

    # Update environment
    cuda_path = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{cuda_major}.{cuda_minor}"
    if os.path.exists(cuda_path):
        os.environ["CUDA_HOME"] = cuda_path
        os.environ["PATH"] = f"{cuda_path}\\bin;" + os.environ.get("PATH", "")

    # Cleanup
    try:
        os.remove(installer_path)
    except:
        pass

    print("[ComfyUI-TRELLIS2] [OK] CUDA compiler installed")
    return True


def setup_cuda_environment():
    """Setup CUDA environment variables for compilation."""
    cuda_home = find_cuda_home()
    if not cuda_home:
        return None

    env = os.environ.copy()
    env["CUDA_HOME"] = cuda_home
    os.environ["CUDA_HOME"] = cuda_home

    cuda_bin = os.path.join(cuda_home, "bin")
    if cuda_bin not in env.get("PATH", ""):
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
        os.environ["PATH"] = env["PATH"]

    return env


def get_cuda_arch_list():
    """Detect GPU compute capability."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}.{minor}"
    except Exception:
        return None


# =============================================================================
# Installation Functions
# =============================================================================

def is_package_installed(import_name):
    """Check if a package is already installed."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def verify_package_import(import_name, package_name):
    """
    Verify that a package can be imported without errors (e.g., ABI mismatch).
    If import fails, uninstall the broken package.
    Returns True if import succeeds, False otherwise.
    """
    try:
        __import__(import_name)
        return True
    except ImportError as e:
        # Check for ABI/symbol errors
        error_str = str(e)
        if "undefined symbol" in error_str or "cannot import" in error_str:
            print(f"[ComfyUI-TRELLIS2] [WARNING] {package_name} has ABI incompatibility: {error_str[:100]}")
            print(f"[ComfyUI-TRELLIS2] Uninstalling broken wheel and will try compilation...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "uninstall", "-y", package_name
                ], capture_output=True, timeout=60)
            except:
                pass
            return False
        # Regular import error - package not installed
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] [WARNING] {package_name} import error: {e}")
        return False


def get_direct_wheel_urls(package_config):
    """
    Build direct wheel URL from GitHub releases.
    Uses static WHEEL_DIRS mapping for reliable URL construction.
    """
    wheel_release_base = package_config.get("wheel_release_base")
    wheel_version = package_config.get("wheel_version")
    wheel_type = package_config.get("wheel_type")

    # flash_attn uses different URL building
    if wheel_type == "flash_attn":
        return []  # Handled by get_flash_attn_wheel_urls

    if not wheel_release_base or not wheel_version:
        return []

    # Use static mapping for wheel directory
    wheel_dir = get_wheel_dir()
    if not wheel_dir:
        return []

    py_major, py_minor = sys.version_info[:2]
    platform = "linux_x86_64" if sys.platform == "linux" else "win_amd64"
    package_name = package_config["name"]

    # Build wheel URL: {release_base}/{wheel_dir}/{package}-{version}-cpXX-cpXX-{platform}.whl
    wheel_name = f"{package_name}-{wheel_version}-cp{py_major}{py_minor}-cp{py_major}{py_minor}-{platform}.whl"
    wheel_url = f"{wheel_release_base}/{wheel_dir}/{wheel_name}"

    return [wheel_url]


def get_flash_attn_wheel_urls():
    """
    Build flash_attn wheel URLs from multiple sources.
    Returns list of (url, version, source_name) tuples to try in order.
    Different repos use different naming conventions.
    """
    torch_ver, cuda_ver = get_torch_info()
    if not torch_ver or not cuda_ver:
        return []

    cuda_mm = cuda_ver.split('.')[0] + cuda_ver.split('.')[1]  # "12.8" -> "128"
    torch_mm = '.'.join(torch_ver.split('.')[:2])  # "2.8.0" -> "2.8"
    py_major, py_minor = sys.version_info[:2]
    platform = "linux_x86_64" if sys.platform == "linux" else "win_amd64"

    urls = []
    for source in FLASH_ATTN_SOURCES:
        if platform not in source["platforms"]:
            continue

        for flash_ver, release_tag in source["versions"]:
            if source["format"] == "cxx11abi":
                # bdashore3/oobabooga format: +cu128torch2.8.0cxx11abiFALSE
                wheel_name = f"flash_attn-{flash_ver}+cu{cuda_mm}torch{torch_mm}.0cxx11abiFALSE-cp{py_major}{py_minor}-cp{py_major}{py_minor}-{platform}.whl"
            else:
                # mjun0812 format: +cu128torch2.8
                wheel_name = f"flash_attn-{flash_ver}+cu{cuda_mm}torch{torch_mm}-cp{py_major}{py_minor}-cp{py_major}{py_minor}-{platform}.whl"

            url = f"{source['base_url']}/{release_tag}/{wheel_name}"
            urls.append((url, flash_ver, source["name"]))

    return urls


def try_install_from_direct_url(package_config):
    """
    Try installing a package from direct GitHub release wheel URLs.
    Tries multiple URL formats (new torch-versioned first, then old).
    Returns True if successful, False otherwise.
    """
    urls = get_direct_wheel_urls(package_config)
    if not urls:
        return False

    package_name = package_config["name"]

    for wheel_url in urls:
        print(f"[ComfyUI-TRELLIS2] Trying: {wheel_url}")

        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", wheel_url
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Verify the package can be imported (check for ABI issues)
                import_name = package_config.get("import_name", package_name)
                if verify_package_import(import_name, package_name):
                    print(f"[ComfyUI-TRELLIS2] [OK] Installed {package_name} from wheel")
                    return True
                else:
                    # ABI mismatch - wheel installed but can't be imported
                    return False
            else:
                # Log the actual error so we can debug
                error_msg = result.stderr[:300] if result.stderr else result.stdout[:300] if result.stdout else "unknown error"
                if "404" in error_msg or "not found" in error_msg.lower() or "no such file" in error_msg.lower():
                    print(f"[ComfyUI-TRELLIS2] Wheel not found at URL")
                    continue
                else:
                    print(f"[ComfyUI-TRELLIS2] pip install failed: {error_msg}")
                    continue
        except subprocess.TimeoutExpired:
            print(f"[ComfyUI-TRELLIS2] Download timed out")
            continue
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] Exception during install: {e}")
            continue

    print(f"[ComfyUI-TRELLIS2] No wheel found at any URL for {package_name}")
    return False


def try_install_from_wheel(package_name, wheel_index_url, import_name=None):
    """
    Try installing a package from pre-built wheels via index.
    Returns True if successful, False otherwise.
    """
    if import_name is None:
        import_name = package_name

    py_ver = get_python_version()

    # Use static mapping for wheel directory
    wheel_dir = get_wheel_dir()
    if not wheel_dir:
        print(f"[ComfyUI-TRELLIS2] No matching wheel directory for {package_name}")
        return False

    # Build the subdirectory URL for this CUDA/torch version
    # e.g., https://pozzettiandrea.github.io/ovoxel-wheels/cu128-torch291/
    wheel_index_with_subdir = f"{wheel_index_url.rstrip('/')}/{wheel_dir}/"

    print(f"[ComfyUI-TRELLIS2] Looking for {package_name} wheel (Python {py_ver}, {wheel_dir})")
    print(f"[ComfyUI-TRELLIS2] Wheel index: {wheel_index_with_subdir}")

    try:
        # Use --no-index to ONLY look at our wheel index, not PyPI
        # This avoids conflicts with similarly-named packages on PyPI
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            package_name, "--no-index", "--find-links", wheel_index_with_subdir
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Verify the package can be imported (check for ABI issues)
            if verify_package_import(import_name, package_name):
                print(f"[ComfyUI-TRELLIS2] [OK] Installed {package_name} from pre-built wheel")
                return True
            else:
                # ABI mismatch - wheel installed but can't be imported
                return False
        else:
            if result.stderr:
                # Check if it's a "no matching distribution" error
                if "No matching distribution" in result.stderr or "no matching distribution" in result.stderr.lower():
                    print(f"[ComfyUI-TRELLIS2] No matching wheel in index for {package_name}")
                else:
                    print(f"[ComfyUI-TRELLIS2] Wheel index install failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[ComfyUI-TRELLIS2] Wheel install timed out for {package_name}")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Wheel install error: {e}")
        return False


def clone_and_init_submodules(git_url, target_dir):
    """
    Clone a git repo and initialize submodules with HTTPS URLs.
    Returns True if successful, False otherwise.
    """
    import tempfile

    # Extract repo URL (remove git+ prefix and #subdirectory suffix)
    repo_url = git_url
    subdirectory = None
    if repo_url.startswith("git+"):
        repo_url = repo_url[4:]
    if "#subdirectory=" in repo_url:
        repo_url, subdirectory = repo_url.split("#subdirectory=")

    print(f"[ComfyUI-TRELLIS2] Cloning {repo_url}...")
    try:
        # Clone the repo
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"[ComfyUI-TRELLIS2] Git clone failed: {result.stderr[:200]}")
            return None

        # Initialize submodules
        print(f"[ComfyUI-TRELLIS2] Initializing submodules...")
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=target_dir, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            # Try to fix SSH URLs to HTTPS
            print(f"[ComfyUI-TRELLIS2] Submodule init failed, trying to fix SSH URLs...")
            gitmodules_path = os.path.join(target_dir, ".gitmodules")
            if os.path.exists(gitmodules_path):
                with open(gitmodules_path, 'r') as f:
                    content = f.read()
                # Replace git@github.com: with https://github.com/
                content = content.replace("git@github.com:", "https://github.com/")
                with open(gitmodules_path, 'w') as f:
                    f.write(content)
                # Sync and retry
                subprocess.run(["git", "submodule", "sync"], cwd=target_dir, capture_output=True)
                result = subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=target_dir, capture_output=True, text=True, timeout=300
                )

        if subdirectory:
            return os.path.join(target_dir, subdirectory)
        return target_dir

    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] Clone error: {e}")
        return None


def try_install_compatible_gcc():
    """
    Try to install a CUDA-compatible GCC version (12 or 13) on Fedora.
    Returns (gcc_path, gxx_path) if successful, (None, None) otherwise.
    """
    if sys.platform != "linux":
        return None, None
    
    distro = detect_linux_distro()
    if distro != "fedora":
        return None, None
    
    # Check if GCC 12 or 13 is already installed
    for gcc_ver in ["12", "13"]:
        gcc_path = shutil.which(f"gcc-{gcc_ver}")
        gxx_path = shutil.which(f"g++-{gcc_ver}")
        if gcc_path and gxx_path:
            return gcc_path, gxx_path
    
    # Try to install GCC 12 or 13
    print("[ComfyUI-TRELLIS2] Attempting to install CUDA-compatible GCC version...")
    print("[ComfyUI-TRELLIS2] This requires sudo access.")
    
    # First, try installing from COPR repository (recommended by RPM Fusion)
    # See: https://rpmfusion.org/Howto/CUDA#GCC_version
    try:
        print("[ComfyUI-TRELLIS2] Trying COPR repository for CUDA-compatible GCC...")
        # Enable COPR repository for CUDA GCC (kwizart/cuda-gcc-10.1)
        # This provides gcc-12/gcc-13 compatible with CUDA
        result = subprocess.run(
            ["sudo", "dnf", "copr", "enable", "-y", "kwizart/cuda-gcc-10.1"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            # Try to install cuda-gcc and cuda-gcc-c++
            for pkg_name in ["cuda-gcc", "cuda-gcc-c++"]:
                install_result = subprocess.run(
                    ["sudo", "dnf", "install", "-y", "-q", pkg_name],
                    capture_output=True, text=True, timeout=300
                )
                if install_result.returncode == 0:
                    # Check if cuda-gcc is available
                    cuda_gcc = shutil.which("cuda-gcc")
                    cuda_gxx = shutil.which("cuda-g++")
                    if cuda_gcc and cuda_gxx:
                        print(f"[ComfyUI-TRELLIS2] [OK] Installed CUDA-compatible GCC from COPR")
                        return cuda_gcc, cuda_gxx
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] COPR repository setup failed: {e}")
    
    # Fallback: Try to install GCC 12 or 13 from standard repositories
    for gcc_ver in ["12", "13"]:
        # Try different package name formats
        package_names = [
            [f"gcc-{gcc_ver}", f"gcc-c++-{gcc_ver}"],
            [f"gcc{gcc_ver}", f"gcc-c++{gcc_ver}"],
        ]
        
        for packages in package_names:
            try:
                # First check if packages are available
                check_result = subprocess.run(
                    ["dnf", "list", "available"] + packages,
                    capture_output=True, text=True, timeout=30
                )
                if all(pkg in check_result.stdout for pkg in packages):
                    # Packages available, try to install
                    result = subprocess.run(
                        ["sudo", "dnf", "install", "-y", "-q"] + packages,
                        capture_output=True, text=True, timeout=300
                    )
                    if result.returncode == 0:
                        gcc_path = shutil.which(f"gcc-{gcc_ver}")
                        gxx_path = shutil.which(f"g++-{gcc_ver}")
                        if gcc_path and gxx_path:
                            print(f"[ComfyUI-TRELLIS2] [OK] Installed and using GCC {gcc_ver}")
                            return gcc_path, gxx_path
            except Exception as e:
                continue
    
    print("[ComfyUI-TRELLIS2] Could not install compatible GCC version")
    print("[ComfyUI-TRELLIS2] You may need to install manually:")
    print("[ComfyUI-TRELLIS2]   sudo dnf copr enable -y kwizart/cuda-gcc-10.1")
    print("[ComfyUI-TRELLIS2]   sudo dnf install -y cuda-gcc cuda-gcc-c++")
    print("[ComfyUI-TRELLIS2] Or install GCC 12/13 from standard repositories if available")
    return None, None


def try_compile_from_source(package_name, git_url):
    """
    Try compiling a package from source.
    Returns True if successful, False otherwise.
    """
    if not git_url:
        print(f"[ComfyUI-TRELLIS2] No source URL available for {package_name}")
        return False

    # Check for CUDA compiler
    cuda_env = setup_cuda_environment()
    if not cuda_env:
        # Try to install CUDA toolkit via pip
        if try_install_cuda_toolkit():
            cuda_env = setup_cuda_environment()

    if not cuda_env:
        print(f"[ComfyUI-TRELLIS2] CUDA compiler not found, cannot compile {package_name}")
        return False

    # Set CUDA architecture
    cuda_arch = get_cuda_arch_list()
    if cuda_arch:
        cuda_env["TORCH_CUDA_ARCH_LIST"] = cuda_arch
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch

    # Platform-specific compiler setup
    if sys.platform == "win32":
        if not shutil.which("cl"):
            print(f"[ComfyUI-TRELLIS2] MSVC C++ compiler not found.")
            print(f"[ComfyUI-TRELLIS2] To install build tools, run: python install_compilers.py")
            print(f"[ComfyUI-TRELLIS2] Or run ComfyUI from 'Developer Command Prompt for VS'")
            return False
    else:
        # Try to find a compatible GCC version
        # CUDA typically supports GCC up to version 12-13, but GCC 15 might be too new
        # Try cuda-gcc (from COPR), then gcc-12, gcc-13, gcc-11, then fallback to system gcc
        gcc = None
        gxx = None
        gcc_version_too_new = False
        
        for gcc_name in ["cuda-gcc", "gcc-12", "gcc-13", "gcc-11", "gcc"]:
            gcc_path = shutil.which(gcc_name)
            if gcc_path:
                gxx_name = gcc_name.replace("gcc", "g++")
                gxx_path = shutil.which(gxx_name)
                if gxx_path:
                    # Check GCC version
                    try:
                        result = subprocess.run([gcc_path, "--version"], capture_output=True, text=True, timeout=5)
                        version_line = result.stdout.split('\n')[0] if result.stdout else ""
                        print(f"[ComfyUI-TRELLIS2] Using {gcc_name}: {version_line[:80]}")
                        # Check if GCC version is too new (15+)
                        if "15." in version_line or "16." in version_line:
                            gcc_version_too_new = True
                            print(f"[ComfyUI-TRELLIS2] [WARNING] GCC 15+ detected. CUDA may not support this version.")
                            # Try to install compatible GCC
                            compatible_gcc, compatible_gxx = try_install_compatible_gcc()
                            if compatible_gcc and compatible_gxx:
                                gcc = compatible_gcc
                                gxx = compatible_gxx
                                print(f"[ComfyUI-TRELLIS2] Using compatible GCC: {gcc}")
                                break
                            else:
                                print(f"[ComfyUI-TRELLIS2] [WARNING] Compilation may fail. Consider installing gcc-12 or gcc-13 manually:")
                                print(f"[ComfyUI-TRELLIS2] [WARNING]   sudo dnf install -y gcc-12 gcc-c++-12")
                                # Still use the too-new GCC as fallback
                                gcc = gcc_path
                                gxx = gxx_path
                        else:
                            gcc = gcc_path
                            gxx = gxx_path
                    except:
                        gcc = gcc_path
                        gxx = gxx_path
                    break
        
        if not gxx:
            print(f"[ComfyUI-TRELLIS2] g++ compiler not found.")
            print(f"[ComfyUI-TRELLIS2] To install build tools, run: python install_compilers.py")
            return False
        if gcc:
            cuda_env["CC"] = gcc
        if gxx:
            cuda_env["CXX"] = gxx
            # If using cuda-gcc, set HOST_COMPILER as recommended by RPM Fusion
            if "cuda-gcc" in gcc or "cuda-g++" in gxx:
                cuda_env["HOST_COMPILER"] = gxx
                os.environ["HOST_COMPILER"] = gxx

    print(f"[ComfyUI-TRELLIS2] Compiling {package_name} from source (this may take several minutes)...")

    # First try direct pip install
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", git_url],
            capture_output=True, text=True, timeout=900, env=cuda_env
        )
        if result.returncode == 0:
            print(f"[ComfyUI-TRELLIS2] [OK] Compiled {package_name} successfully")
            return True
    except:
        pass

    # If direct install fails, try manual clone with submodule fix
    print(f"[ComfyUI-TRELLIS2] Direct install failed, trying manual clone with submodule fix...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "repo")
        source_path = clone_and_init_submodules(git_url, clone_dir)

        if source_path and os.path.exists(source_path):
            # Install any missing pure-Python dependencies first
            print(f"[ComfyUI-TRELLIS2] Installing build dependencies...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "plyfile", "zstandard", "ninja"],
                capture_output=True, text=True, timeout=120
            )

            # Try to install with --no-deps to skip problematic dependencies
            print(f"[ComfyUI-TRELLIS2] Building from cloned source...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", source_path],
                capture_output=True, text=True, timeout=900, env=cuda_env
            )
            if result.returncode == 0:
                print(f"[ComfyUI-TRELLIS2] [OK] Compiled {package_name} successfully")
                return True
            else:
                print(f"[ComfyUI-TRELLIS2] Compilation failed for {package_name}")
                if result.stderr:
                    # Show more of the error (up to 2000 chars) to help diagnose issues
                    error_output = result.stderr
                    print(f"[ComfyUI-TRELLIS2] Error output ({len(error_output)} chars):")
                    print(error_output[:2000])
                    if len(error_output) > 2000:
                        print(f"[ComfyUI-TRELLIS2] ... (truncated, {len(error_output) - 2000} more chars)")
                if result.stdout:
                    # Also check stdout for errors
                    stdout_output = result.stdout
                    if "error" in stdout_output.lower() or "failed" in stdout_output.lower():
                        print(f"[ComfyUI-TRELLIS2] Error in stdout:")
                        print(stdout_output[-1000:])  # Last 1000 chars often contain the actual error
                return False

    print(f"[ComfyUI-TRELLIS2] Compilation failed for {package_name}")
    return False


def try_install_flash_attn():
    """
    Try installing flash_attn from pre-built wheels.
    Tries multiple sources (bdashore3, mjun0812, oobabooga) until one works.
    Returns True if successful, False otherwise.
    """
    urls = get_flash_attn_wheel_urls()
    if not urls:
        print("[ComfyUI-TRELLIS2] Could not determine flash_attn wheel URL for this environment")
        return False

    for url, version, source in urls:
        print(f"[ComfyUI-TRELLIS2] Trying flash_attn {version} from {source}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", url
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"[ComfyUI-TRELLIS2] [OK] Installed flash_attn {version} from {source}")
                return True
            else:
                # Check for 404 or other errors
                error_msg = result.stderr or result.stdout or ""
                if "404" in error_msg or "not found" in error_msg.lower():
                    continue  # Try next URL
                else:
                    # Log non-404 errors but continue trying
                    print(f"[ComfyUI-TRELLIS2] {source} failed: {error_msg[:100]}")
                    continue
        except subprocess.TimeoutExpired:
            print(f"[ComfyUI-TRELLIS2] {source} timed out")
            continue
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] {source} error: {e}")
            continue

    print("[ComfyUI-TRELLIS2] Could not find flash_attn wheel from any source")
    return False


def install_cuda_package(package_config):
    """
    Install a CUDA package - tries wheel first, falls back to compilation.
    Returns True if successful, False otherwise.
    """
    name = package_config["name"]
    import_name = package_config["import_name"]
    wheel_index = package_config.get("wheel_index")
    git_url = package_config.get("git_url")
    wheel_type = package_config.get("wheel_type")

    print(f"\n[ComfyUI-TRELLIS2] Installing {name}...")

    # Always reinstall to ensure users get the latest fixed wheels
    # (Previous wheel builds had naming bugs that caused installation failures)

    # Check if we have CUDA
    torch_ver, cuda_ver = get_torch_info()
    if not cuda_ver:
        print(f"[ComfyUI-TRELLIS2] [SKIP] PyTorch CUDA not available, skipping {name}")
        return False

    # Special handling for flash_attn
    if wheel_type == "flash_attn":
        if try_install_flash_attn():
            return True
        # Fall through to compilation
    else:
        # Try 1: direct GitHub release URL (most reliable - bypasses pip index parsing)
        if try_install_from_direct_url(package_config):
            return True

        # Try 2: wheel index as fallback (pip --find-links)
        if wheel_index and try_install_from_wheel(name, wheel_index, import_name):
            return True

    # Try 3: compile from source
    print(f"[ComfyUI-TRELLIS2] No pre-built wheel found, attempting compilation...")
    if try_compile_from_source(name, git_url):
        return True

    print(f"[ComfyUI-TRELLIS2] [FAILED] Could not install {name}")
    return False


def install_requirements():
    """Install dependencies from requirements.txt."""
    print("[ComfyUI-TRELLIS2] Installing requirements.txt dependencies...")

    script_dir = Path(__file__).parent.absolute()
    requirements_path = script_dir / "requirements.txt"

    if not requirements_path.exists():
        print("[ComfyUI-TRELLIS2] [WARNING] requirements.txt not found")
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            capture_output=True, text=True, timeout=600
        )

        if result.returncode == 0:
            print("[ComfyUI-TRELLIS2] [OK] Requirements installed successfully")
            return True
        else:
            print("[ComfyUI-TRELLIS2] [WARNING] Some requirements failed to install")
            if result.stderr:
                print(f"[ComfyUI-TRELLIS2] Error: {result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print("[ComfyUI-TRELLIS2] [WARNING] Requirements installation timed out")
        return False
    except Exception as e:
        print(f"[ComfyUI-TRELLIS2] [WARNING] Requirements error: {e}")
        return False


def try_install_python_headers():
    """
    Install Python development headers for embedded Python on Windows.
    Required for Triton to compile CUDA utilities at runtime.

    Downloads include/ and libs/ folders from triton-windows releases.
    """
    # Only needed on Windows
    if sys.platform != "win32":
        return True

    # Check if we're using embedded Python (ComfyUI portable)
    python_path = Path(sys.executable)
    python_dir = python_path.parent

    # Look for python_embeded folder pattern
    if "python_embeded" not in str(python_dir).lower() and "python_embedded" not in str(python_dir).lower():
        # Not embedded Python, probably has headers already
        print("[ComfyUI-TRELLIS2] [SKIP] Not using embedded Python, headers should exist")
        return True

    include_dir = python_dir / "include"
    libs_dir = python_dir / "libs"

    # Check if Python.h already exists
    python_h = include_dir / "Python.h"
    if python_h.exists():
        print("[ComfyUI-TRELLIS2] [OK] Python headers already installed")
        return True

    print("[ComfyUI-TRELLIS2] Installing Python headers for Triton compatibility...")
    print(f"[ComfyUI-TRELLIS2] Target directory: {python_dir}")

    # Determine Python version for download URL
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Try version-specific URL first, fall back to 3.12.7 (most common for ComfyUI)
    urls_to_try = [
        f"https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_{py_version}_include_libs.zip",
        "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip",
    ]

    for url in urls_to_try:
        try:
            print(f"[ComfyUI-TRELLIS2] Downloading from {url}...")

            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / "python_headers.zip"

                # Download the zip file
                urllib.request.urlretrieve(url, zip_path)
                print("[ComfyUI-TRELLIS2] Download complete, extracting...")

                # Extract to python_embeded directory
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(python_dir)

                # Verify extraction
                if python_h.exists():
                    print("[ComfyUI-TRELLIS2] [OK] Python headers installed successfully")
                    return True
                else:
                    print("[ComfyUI-TRELLIS2] [WARNING] Extraction completed but Python.h not found")

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"[ComfyUI-TRELLIS2] [INFO] Headers for Python {py_version} not available, trying fallback...")
                continue
            print(f"[ComfyUI-TRELLIS2] [WARNING] Download failed: {e}")
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] [WARNING] Failed to install Python headers: {e}")

    print("[ComfyUI-TRELLIS2] [WARNING] Could not install Python headers")
    print("[ComfyUI-TRELLIS2] [WARNING] Triton may fail to compile CUDA utilities")
    print("[ComfyUI-TRELLIS2] [WARNING] Manual fix: download Python include/libs folders from")
    print("[ComfyUI-TRELLIS2] [WARNING] https://github.com/woct0rdho/triton-windows/releases")
    return False


def main():
    print("\n" + "=" * 60)
    print("ComfyUI-TRELLIS2 Installation")
    print("=" * 60)

    # Show detected environment
    py_ver = get_python_version()
    torch_ver, cuda_ver = get_torch_info()

    print(f"[ComfyUI-TRELLIS2] Python: {py_ver}")
    if torch_ver:
        print(f"[ComfyUI-TRELLIS2] PyTorch: {torch_ver}")
    if cuda_ver:
        print(f"[ComfyUI-TRELLIS2] CUDA: {cuda_ver}")

    # Warn about Python version compatibility
    if sys.version_info >= (3, 13):
        print(f"[ComfyUI-TRELLIS2] [WARNING] Python {py_ver} detected - pre-built wheels may not be available")
        print(f"[ComfyUI-TRELLIS2] [WARNING] Will attempt compilation from source where possible")

    wheel_suffix = get_wheel_cuda_suffix()
    if wheel_suffix:
        print(f"[ComfyUI-TRELLIS2] Wheel suffix: {wheel_suffix}")

    # Install Visual C++ Redistributable on Windows (required for opencv, etc.)
    if sys.platform == "win32":
        try_install_vcredist()

    # Install triton-windows on Windows (required by flex_gemm)
    if sys.platform == "win32":
        print("\n[ComfyUI-TRELLIS2] Installing triton-windows (required for flex_gemm)...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "triton-windows"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print("[ComfyUI-TRELLIS2] [OK] triton-windows installed")
            else:
                print(f"[ComfyUI-TRELLIS2] [WARNING] triton-windows install failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
        except Exception as e:
            print(f"[ComfyUI-TRELLIS2] [WARNING] triton-windows install error: {e}")

        # Install Python headers for embedded Python (required for Triton to compile CUDA utils)
        try_install_python_headers()

    # Install requirements.txt first
    print("\n" + "-" * 60)
    install_requirements()

    # Install CUDA packages
    results = {}
    for pkg in PACKAGES:
        print("-" * 60)
        results[pkg["name"]] = install_cuda_package(pkg)

    # Summary
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"  {name}: {status}")
    print("=" * 60)

    # Overall status
    if all(results.values()):
        print("[ComfyUI-TRELLIS2] Installation completed successfully!")
    elif any(results.values()):
        print("[ComfyUI-TRELLIS2] Installation completed with some failures")
        print("[ComfyUI-TRELLIS2] TRELLIS2 may still work with reduced functionality")
    else:
        print("[ComfyUI-TRELLIS2] Installation failed")
        print("[ComfyUI-TRELLIS2] Check that you have PyTorch with CUDA support installed")


if __name__ == "__main__":
    main()
