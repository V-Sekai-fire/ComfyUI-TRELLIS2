# ComfyUI-TRELLIS2

ComfyUI custom nodes for [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) - Microsoft's state-of-the-art image-to-3D generation model.

Generate high-quality 3D meshes with PBR (Physically Based Rendering) materials from a single image.

### Install via ComfyUI Manager (Recommended)

Search for "ComfyUI-TRELLIS2" in ComfyUI Manager and click install.

## Example Workflow

![tpose](docs/tpose.png)

![rmbg](docs/rmbg.png)


https://github.com/user-attachments/assets/e28e4a74-b119-4303-8e30-63361f26aa88


## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU with 8GB VRAM (16GB+ recommended)

All dependencies should theoretically install through wheel by running install.py!
This node was made for comfyui-manager install, but you can also pip install requirements and "python install.py" manually.

## Notes

The video render nodes will be implemented soon! Not yet working.

## Manual Model Download

Models are downloaded automatically from HuggingFace on first run. If you need to download manually:

**From [microsoft/TRELLIS-image-large/ckpts](https://huggingface.co/microsoft/TRELLIS-image-large/tree/main/ckpts):**
- `ss_dec_conv3d_16l8_fp16.json` + `ss_dec_conv3d_16l8_fp16.safetensors`

**From [microsoft/TRELLIS.2-4B/ckpts](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/main/ckpts):**
- `ss_flow_img_dit_1_3B_64_bf16.json` + `.safetensors`
- `shape_dec_next_dc_f16c32_fp16.json` + `.safetensors`
- `slat_flow_img2shape_dit_1_3B_512_bf16.json` + `.safetensors`
- `slat_flow_img2shape_dit_1_3B_1024_bf16.json` + `.safetensors`
- `tex_dec_next_dc_f16c32_fp16.json` + `.safetensors`
- `slat_flow_imgshape2tex_dit_1_3B_512_bf16.json` + `.safetensors`
- `slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json` + `.safetensors`

**Place files in:** `ComfyUI/models/trellis2/ckpts/`

```text
ComfyUI/models/trellis2/
└── ckpts/
    ├── ss_dec_conv3d_16l8_fp16.json
    ├── ss_dec_conv3d_16l8_fp16.safetensors
    ├── ss_flow_img_dit_1_3B_64_bf16.json
    ├── ss_flow_img_dit_1_3B_64_bf16.safetensors
    ├── shape_dec_next_dc_f16c32_fp16.json
    ├── shape_dec_next_dc_f16c32_fp16.safetensors
    ├── ... (etc)
```

**Other HuggingFace models (auto-downloaded):**
- DinoV3: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- BiRefNet: `briaai/RMBG-2.0`

### Manual Installation via Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2.git
cd ComfyUI-TRELLIS2
pip install -r requirements.txt
python install.py
```

Or if you want to use the forked version with bug fixes:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/V-Sekai-fire/ComfyUI-TRELLIS2.git
cd ComfyUI-TRELLIS2
pip install -r requirements.txt
python install.py
```

## Recent Bug Fixes

This fork includes several important bug fixes:

### 1. CUDA Kernel Error Fix for `torch.segment_reduce`
**Issue:** `torch.segment_reduce` does not have CUDA kernel support for all operations, causing `RuntimeError: DispatchStub: missing kernel for cuda` errors.

**Fix:** Modified `trellis2/modules/sparse/basic.py` to automatically use CPU for `segment_reduce` operations when on CUDA devices, then move results back to GPU.

**Files Changed:**
- `trellis2/modules/sparse/basic.py`

### 2. Pipeline Type Mismatch Fix
**Issue:** When loading models with resolution "1024" (non-cascade), the code was incorrectly trying to use "1024_cascade" pipeline type, which requires both 512 and 1024 models.

**Fix:** Modified `nodes/nodes_inference.py` to use the pipeline's `default_pipeline_type` (set when model was loaded) instead of hardcoded mapping, with automatic fallback to non-cascade when cascade models aren't available.

**Files Changed:**
- `nodes/nodes_inference.py`

### 3. OutOfMemoryError Handling Improvements
**Issue:** When `OutOfMemoryError` occurred during model weight loading, the exception handler was retrying the entire model loading process, causing re-downloads with broken paths and empty filenames.

**Fix:** 
- Moved `OutOfMemoryError` handling to the weight loading stage in `trellis2/models/__init__.py`
- Only retries weight loading (not config downloading or path parsing)
- Clears GPU cache and runs garbage collection before retry
- Added path validation to prevent downloading with empty model names

**Files Changed:**
- `trellis2/models/__init__.py`
- `trellis2/pipelines/base.py`

### 4. Path Parsing and Validation
**Issue:** Invalid HuggingFace paths could cause empty `model_name` errors when trying to download config files.

**Fix:**
- Added validation to ensure HuggingFace paths have correct format (at least 3 parts: `repo_owner/repo_name/model/path`)
- Added validation to check `model_name` is not empty before attempting download
- Improved path handling for both relative and absolute HuggingFace paths

**Files Changed:**
- `trellis2/models/__init__.py`
- `trellis2/pipelines/base.py`

### 5. Line Ending Normalization
**Issue:** Mixed line endings (CRLF/LF) causing git warnings.

**Fix:** Added `.gitattributes` file to normalize line endings to LF for all text files.

**Files Changed:**
- `.gitattributes` (new file)

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

Thank you @fire for pointing out the segment_reduce bug!
