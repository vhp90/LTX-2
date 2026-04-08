#!/bin/bash
# ==============================================================================
#  LTX-2 Studio — Full Environment Setup
#  Smart install: skips packages already at the required version.
#  Maximizes speed: parallel downloads, prebuilt wheels, all CPU cores for builds.
#
#  Usage:
#    bash setup_environment.sh              # Full setup
#    bash setup_environment.sh --skip-models  # Skip model downloads
# ==============================================================================

SKIP_MODELS=false
for arg in "$@"; do
  case $arg in --skip-models) SKIP_MODELS=true ;; esac
done

# Use all CPU cores for any source builds (flash-attn etc.)
export MAX_JOBS=$(nproc)
export MAKEFLAGS="-j$(nproc)"

echo ""
echo "========================================"
echo "  LTX-2 Studio — Environment Setup"
echo "========================================"
echo ""

# --------------------------------------------------------------------------
#  Helpers — defined at top level so they're available everywhere
# --------------------------------------------------------------------------

# Returns 0 if package is installed and meets optional min version
pip_installed() {
  local pkg="$1" min_ver="$2"
  local installed
  installed=$(pip show "$pkg" 2>/dev/null | grep -i "^Version:" | awk '{print $2}')
  [ -z "$installed" ] && return 1
  if [ -n "$min_ver" ]; then
    python -c "from packaging.version import Version; exit(0 if Version('$installed') >= Version('$min_ver') else 1)" 2>/dev/null
    return $?
  fi
  return 0
}

# pip_smart_install <label> <check_pkg> <min_ver_or_empty> <pip_args...>
# Skips if already installed at required version, otherwise installs.
pip_smart_install() {
  local label="$1" check_pkg="$2" min_ver="$3"
  shift 3
  if pip_installed "$check_pkg" "$min_ver"; then
    local ver
    ver=$(pip show "$check_pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
    echo "  ⏭  $label ($ver) — already installed"
    return 0
  fi
  echo "  ⬇  $label..."
  pip install --prefer-binary "$@" 2>&1 | tail -1
  echo "  ✅ $label done"
}

# --------------------------------------------------------------------------
#  1. System info
# --------------------------------------------------------------------------
echo "[1/6] System info"
echo "  Python:  $(python --version 2>&1)"
echo "  CUDA:    $(nvcc --version 2>&1 | grep release | awk '{print $6}' || echo 'N/A')"
echo "  CPUs:    $(nproc) cores"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "N/A")
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "N/A")
  echo "  GPU:     $GPU_NAME ($GPU_MEM)"
else
  echo "  GPU:     Not detected"
fi
echo ""

# Ensure packaging module is available for version comparisons
python -c "import packaging" 2>/dev/null || pip install packaging -q

# --------------------------------------------------------------------------
#  2. PyTorch + Triton
# --------------------------------------------------------------------------
echo "[2/6] PyTorch + Triton"

pip_smart_install "torch 2.7+ (cu130)" "torch" "2.7.0" \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch torchaudio torchvision

pip_smart_install "triton" "triton" "" triton

echo ""

# --------------------------------------------------------------------------
#  3. Attention backends
# --------------------------------------------------------------------------
echo "[3/6] Attention backends"

# Detect GPU compute capability to decide what to install
GPU_CAP=$(python -c "import torch; print(torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")

if [ "$GPU_CAP" -ge 10 ]; then
  echo "  ℹ️  Blackwell GPU (sm_${GPU_CAP}x) — using PyTorch native SDPA (cuDNN + Flash backends)"
  echo "  ℹ️  xformers skipped (no Blackwell support yet)"
  # Remove xformers if it was previously installed
  pip uninstall xformers -y > /dev/null 2>&1 || true
else
  pip_smart_install "xformers" "xformers" "" \
    --index-url https://download.pytorch.org/whl/cu130 \
    xformers
fi

# flash-attn: skip on Blackwell — PyTorch native cuDNN SDPA is already fastest
if [ "$GPU_CAP" -ge 10 ]; then
  echo "  ℹ️  flash-attn skipped — PyTorch cuDNN SDPA is faster on Blackwell"
elif pip_installed "flash-attn"; then
  ver=$(pip show flash-attn 2>/dev/null | grep "^Version:" | awk '{print $2}')
  echo "  ⏭  flash-attn ($ver) — already installed"
else
  echo "  ⬇  flash-attn (trying prebuilt wheel only, no source build)..."
  TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
  PY_VER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  PREBUILT_URL="https://github.com/Dao-AILab/flash-attention/releases/latest/download/flash_attn-${TORCH_VER}-${PY_VER}-${PY_VER}-linux_x86_64.whl"
  pip install "$PREBUILT_URL" 2>/dev/null \
    || echo "  ℹ️  No prebuilt flash-attn wheel — PyTorch SDPA will be used"
fi

echo ""

# --------------------------------------------------------------------------
#  4. Python dependencies + local packages
# --------------------------------------------------------------------------
echo "[4/6] Python dependencies"

pip_smart_install "einops"           "einops"           ""     einops
pip_smart_install "numpy"            "numpy"            ""     numpy
pip_smart_install "transformers"     "transformers"     "4.52" "transformers>=4.52"
pip_smart_install "safetensors"      "safetensors"      ""     safetensors
pip_smart_install "accelerate"       "accelerate"       ""     accelerate
pip_smart_install "scipy"            "scipy"            "1.14" "scipy>=1.14"
pip_smart_install "av"               "av"               ""     av
pip_smart_install "tqdm"             "tqdm"             ""     tqdm
pip_smart_install "pillow"           "pillow"           ""     pillow
pip_smart_install "fastapi"          "fastapi"          ""     fastapi
pip_smart_install "uvicorn"          "uvicorn"          ""     "uvicorn[standard]"
pip_smart_install "pyyaml"           "pyyaml"           ""     pyyaml
pip_smart_install "huggingface_hub"  "huggingface_hub"  ""     huggingface_hub
pip_smart_install "pydantic"         "pydantic"         ""     pydantic
pip_smart_install "python-multipart" "python-multipart" ""     python-multipart

# Local editable packages
install_local() {
  local label="$1" path="$2" pkg="$3"
  if pip_installed "$pkg"; then
    local loc
    loc=$(pip show "$pkg" 2>/dev/null | grep "^Editable project location:" | awk '{print $NF}')
    if [ -n "$loc" ]; then
      echo "  ⏭  $label (editable) — already installed"
      return 0
    fi
  fi
  echo "  ⬇  $label (editable)..."
  pip install -e "$path" 2>&1 | tail -1
  echo "  ✅ $label done"
}

install_local "ltx-core"      "packages/ltx-core"      "ltx-core"
install_local "ltx-pipelines" "packages/ltx-pipelines" "ltx-pipelines"

echo ""

# --------------------------------------------------------------------------
#  5. Frontend (npm)
# --------------------------------------------------------------------------
echo "[5/6] Frontend dependencies"

if [ -d "app/frontend/node_modules" ]; then
  if [ "app/frontend/package.json" -nt "app/frontend/node_modules" ]; then
    echo "  ⬇  package.json changed — updating..."
    npm --prefix app/frontend install > /dev/null 2>&1
    echo "  ✅ Frontend updated"
  else
    echo "  ⏭  node_modules up to date — skipping"
  fi
else
  echo "  ⬇  Installing frontend dependencies..."
  npm --prefix app/frontend install > /dev/null 2>&1
  echo "  ✅ Frontend installed"
fi

echo ""

# --------------------------------------------------------------------------
#  6. Models
# --------------------------------------------------------------------------
if [ "$SKIP_MODELS" = true ]; then
  echo "[6/6] Skipping model downloads (--skip-models)"
else
  echo "[6/6] Syncing models from config.yaml..."
  python -m app.setup_models
fi

echo ""
echo "========================================"
echo "  Setup complete. Run ./start.sh"
echo "========================================"
