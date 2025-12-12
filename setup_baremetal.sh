#!/bin/bash
# Copyright (c) 2025 MLPerf Training
# Setup script for baremetal Flux training on AMD GPUs with ROCm

set -e

# =============================================================================
# Configuration
# =============================================================================

# Default paths (can be overridden by environment variables)
DATAROOT=${DATAROOT:-"./datasets"}
MODELROOT=${MODELROOT:-"./models"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"flux-mlperf"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}

# Installation options
INSTALL_DEPS=${INSTALL_DEPS:-true}
CREATE_VENV=${CREATE_VENV:-true}
DOWNLOAD_PREPROCESSED=${DOWNLOAD_PREPROCESSED:-true}
DOWNLOAD_CC12M=${DOWNLOAD_CC12M:-false}  # Off by default - CC12M is ~2.5TB!
DOWNLOAD_RAW=${DOWNLOAD_RAW:-false}
DOWNLOAD_ENCODERS=${DOWNLOAD_ENCODERS:-false}

# ROCm version for PyTorch installation
# Options: "auto" (default), "latest", "6.2", "6.3", "6.4", "nightly"
ROCM_VERSION=${ROCM_VERSION:-"auto"}

# PyTorch ROCm wheel URLs
PYTORCH_ROCM_6_2_URL="https://download.pytorch.org/whl/rocm6.2"
PYTORCH_ROCM_6_3_URL="https://download.pytorch.org/whl/rocm6.3"
PYTORCH_ROCM_6_4_URL="https://download.pytorch.org/whl/rocm6.4"
PYTORCH_ROCM_NIGHTLY_URL="https://download.pytorch.org/whl/nightly/rocm7.1"

# Auto-detect ROCm version from system
detect_rocm_version() {
    local rocm_version=""
    local source=""
    
    # 1. Check CRAY_ROCM_VERSION (Cray/HPE systems with module loaded)
    if [[ -n "$CRAY_ROCM_VERSION" ]]; then
        rocm_version=$(echo "$CRAY_ROCM_VERSION" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        source="CRAY_ROCM_VERSION env var: $CRAY_ROCM_VERSION"
    fi
    
    # 2. Check ROCM_VERSION environment variable (generic modules)
    if [[ -z "$rocm_version" && -n "$ROCM_VER" ]]; then
        rocm_version=$(echo "$ROCM_VER" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        source="ROCM_VER env var: $ROCM_VER"
    fi
    
    # 3. Check ROCM_PATH environment variable (set by modules)
    if [[ -z "$rocm_version" && -n "$ROCM_PATH" ]]; then
        rocm_version=$(echo "$ROCM_PATH" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        source="ROCM_PATH env var: $ROCM_PATH"
    fi
    
    # 3. Check loaded modules for rocm
    if [[ -z "$rocm_version" ]] && command -v module &> /dev/null; then
        # Try to get loaded rocm module
        local loaded_rocm=$(module list 2>&1 | grep -iE 'rocm[/-]' | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        if [[ -n "$loaded_rocm" ]]; then
            rocm_version="$loaded_rocm"
            source="loaded module (module list)"
        fi
    fi
    
    # 4. Check HIP_PATH (another common module variable)
    if [[ -z "$rocm_version" && -n "$HIP_PATH" ]]; then
        rocm_version=$(echo "$HIP_PATH" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        source="HIP_PATH env var: $HIP_PATH"
    fi
    
    # 5. Try hipcc --version (uses the module's hipcc if loaded)
    if [[ -z "$rocm_version" ]] && command -v hipcc &> /dev/null; then
        local hipcc_output=$(hipcc --version 2>&1)
        rocm_version=$(echo "$hipcc_output" | grep -iE '(HIP version|hip version)' | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        source="hipcc --version"
    fi
    
    # 6. Try ROCM_PATH/.info/version if ROCM_PATH is set
    if [[ -z "$rocm_version" && -n "$ROCM_PATH" && -f "$ROCM_PATH/.info/version" ]]; then
        local file_content=$(cat "$ROCM_PATH/.info/version" 2>/dev/null)
        rocm_version=$(echo "$file_content" | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        source="$ROCM_PATH/.info/version"
    fi
    
    # 7. Fallback: Try /opt/rocm/.info/version (system default, may not match module)
    if [[ -z "$rocm_version" && -f /opt/rocm/.info/version ]]; then
        local file_content=$(cat /opt/rocm/.info/version 2>/dev/null)
        rocm_version=$(echo "$file_content" | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        source="/opt/rocm/.info/version (system default - may not match loaded module!)"
    fi
    
    # 8. Fallback: rocm-smi --version  
    if [[ -z "$rocm_version" ]] && command -v rocm-smi &> /dev/null; then
        rocm_version=$(rocm-smi --version 2>&1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
        source="rocm-smi --version"
    fi
    
    # Output source for debugging (to stderr)
    if [[ -n "$source" ]]; then
        echo "  Detection source: $source" >&2
    fi
    
    # Return just the version
    echo "$rocm_version"
}

# Map detected ROCm version to PyTorch wheel version
map_rocm_to_pytorch() {
    local detected="$1"
    local major_minor=$(echo "$detected" | grep -oP '^\d+\.\d+')
    
    case "$major_minor" in
        6.2*)
            echo "6.2"
            ;;
        6.3*)
            echo "6.3"
            ;;
        6.4*)
            echo "6.4"
            ;;
        7.*)
            echo "nightly"
            ;;
        *)
            # Default to latest stable if unknown
            echo "6.4"
            ;;
    esac
}

# HuggingFace token (required only for encoder download)
HF_TOKEN=${HF_TOKEN:-""}

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
}

print_step() {
    echo ""
    echo ">>> $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed or not in PATH"
        return 1
    fi
    return 0
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup script for baremetal Flux training on AMD GPUs with ROCm.

Options:
    -h, --help              Show this help message
    -d, --dataroot PATH     Path to store datasets (default: ./datasets)
    -m, --modelroot PATH    Path to store model encoders (default: ./models)
    -e, --env-name NAME     Conda environment name (default: flux-mlperf)
    --python VERSION        Python version for conda env (default: 3.11)
    --no-env                Skip conda environment creation
    --no-deps               Skip dependency installation
    --no-download           Skip all data downloads
    --download-cc12m        Download CC12M training dataset (~2.5TB) - OFF by default
    --download-raw          Download raw (non-preprocessed) datasets
    --download-encoders     Download model encoders (requires --hf-token)
    --hf-token TOKEN        HuggingFace token for encoder download
    --all                   Download everything (preprocessed + CC12M, raw, encoders)
    --rocm-version VERSION  ROCm version for PyTorch (default: auto-detect)

ROCm Version Options:
    auto                    Auto-detect ROCm version from system - DEFAULT
    latest                  Install PyTorch for ROCm 6.4 (latest stable)
    6.2                     Install PyTorch for ROCm 6.2.x
    6.3                     Install PyTorch for ROCm 6.3.x
    6.4                     Install PyTorch for ROCm 6.4.x
    nightly                 Install PyTorch nightly for ROCm 7.1 (bleeding edge)

Environment Variables:
    DATAROOT                Same as --dataroot
    MODELROOT               Same as --modelroot
    CONDA_ENV_NAME          Same as --env-name
    PYTHON_VERSION          Same as --python
    HF_TOKEN                Same as --hf-token
    ROCM_VERSION            Same as --rocm-version

Examples:
    # Basic setup - downloads validation data only (COCO + empty_encodings)
    # CC12M training data (~2.5TB) is NOT downloaded by default
    $0 --dataroot /path/to/data --modelroot /path/to/models

    # Include CC12M training data download (~2.5TB)
    $0 --download-cc12m --dataroot /path/to/data --modelroot /path/to/models

    # Custom conda environment name
    $0 --env-name my-flux-env --dataroot /path/to/data --modelroot /path/to/models

    # Force specific ROCm version (overrides auto-detect)
    $0 --rocm-version 6.2 --dataroot /path/to/data --modelroot /path/to/models

    # Setup with nightly PyTorch (ROCm 7.1)
    $0 --rocm-version nightly --dataroot /path/to/data --modelroot /path/to/models

    # Setup without conda environment (use current env)
    $0 --no-env --dataroot /path/to/data

    # Full setup including CC12M, raw data, and encoders
    $0 --all --hf-token <your_token> --dataroot /path/to/data

EOF
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -d|--dataroot)
            DATAROOT="$2"
            shift 2
            ;;
        -m|--modelroot)
            MODELROOT="$2"
            shift 2
            ;;
        -e|--env-name)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --no-env)
            CREATE_VENV=false
            shift
            ;;
        --no-deps)
            INSTALL_DEPS=false
            shift
            ;;
        --no-download)
            DOWNLOAD_PREPROCESSED=false
            DOWNLOAD_CC12M=false
            DOWNLOAD_RAW=false
            DOWNLOAD_ENCODERS=false
            shift
            ;;
        --download-cc12m)
            DOWNLOAD_CC12M=true
            shift
            ;;
        --download-raw)
            DOWNLOAD_RAW=true
            shift
            ;;
        --download-encoders)
            DOWNLOAD_ENCODERS=true
            shift
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --all)
            DOWNLOAD_PREPROCESSED=true
            DOWNLOAD_CC12M=true
            DOWNLOAD_RAW=true
            DOWNLOAD_ENCODERS=true
            shift
            ;;
        --rocm-version)
            ROCM_VERSION="$2"
            if [[ "$ROCM_VERSION" != "auto" && "$ROCM_VERSION" != "latest" && "$ROCM_VERSION" != "6.2" && "$ROCM_VERSION" != "6.3" && "$ROCM_VERSION" != "6.4" && "$ROCM_VERSION" != "nightly" ]]; then
                echo "Error: Invalid ROCm version '$ROCM_VERSION'. Use 'auto', 'latest', '6.2', '6.3', '6.4', or 'nightly'"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validation
# =============================================================================

print_header "Flux Baremetal Setup for AMD GPUs (ROCm)"

# Handle auto-detection of ROCm version
DETECTED_ROCM=""
if [[ "$ROCM_VERSION" == "auto" ]]; then
    print_step "Auto-detecting ROCm version..."
    
    # Capture version (stdout) and show debug info (stderr)
    DETECTED_ROCM=$(detect_rocm_version)
    
    if [[ -n "$DETECTED_ROCM" ]]; then
        echo "  Detected: $DETECTED_ROCM"
        ROCM_VERSION=$(map_rocm_to_pytorch "$DETECTED_ROCM")
        echo "  PyTorch wheel: rocm$ROCM_VERSION"
    else
        echo "  Warning: Could not detect ROCm version"
        echo "  Defaulting to latest stable (6.4)"
        ROCM_VERSION="6.4"
    fi
    echo ""
    echo "  If incorrect, re-run with: --rocm-version <version>"
    echo "  Valid versions: 6.2, 6.3, 6.4, nightly"
fi

# Determine PyTorch ROCm URL based on version selection
PYTORCH_PRE_FLAG=""
case "$ROCM_VERSION" in
    6.2)
        PYTORCH_ROCM_URL="$PYTORCH_ROCM_6_2_URL"
        ROCM_VERSION_DISPLAY="6.2"
        ;;
    6.3)
        PYTORCH_ROCM_URL="$PYTORCH_ROCM_6_3_URL"
        ROCM_VERSION_DISPLAY="6.3"
        ;;
    6.4|latest)
        PYTORCH_ROCM_URL="$PYTORCH_ROCM_6_4_URL"
        ROCM_VERSION_DISPLAY="6.4 (latest stable)"
        ;;
    nightly)
        PYTORCH_ROCM_URL="$PYTORCH_ROCM_NIGHTLY_URL"
        PYTORCH_PRE_FLAG="--pre"
        ROCM_VERSION_DISPLAY="nightly (ROCm 7.1)"
        ;;
esac

if [[ -n "$DETECTED_ROCM" ]]; then
    ROCM_VERSION_DISPLAY="$ROCM_VERSION_DISPLAY (detected: $DETECTED_ROCM)"
fi

echo "Configuration:"
echo "  DATAROOT:             $DATAROOT"
echo "  MODELROOT:            $MODELROOT"
echo "  CONDA_ENV_NAME:       $CONDA_ENV_NAME"
echo "  PYTHON_VERSION:       $PYTHON_VERSION"
echo "  CREATE_ENV:           $CREATE_VENV"
echo "  INSTALL_DEPS:         $INSTALL_DEPS"
echo "  ROCM_VERSION:         $ROCM_VERSION_DISPLAY"
echo "  PYTORCH_URL:          $PYTORCH_ROCM_URL"
echo "  DOWNLOAD_PREPROCESSED: $DOWNLOAD_PREPROCESSED (COCO + empty_encodings)"
echo "  DOWNLOAD_CC12M:       $DOWNLOAD_CC12M (~2.5TB training data)"
echo "  DOWNLOAD_RAW:         $DOWNLOAD_RAW"
echo "  DOWNLOAD_ENCODERS:    $DOWNLOAD_ENCODERS"

if [[ "$DOWNLOAD_ENCODERS" == "true" && -z "$HF_TOKEN" ]]; then
    echo ""
    echo "Warning: --download-encoders requires --hf-token"
    echo "Encoder download will be skipped."
    DOWNLOAD_ENCODERS=false
fi

# Check for required commands
print_step "Checking prerequisites..."
check_command python3 || exit 1
check_command pip || exit 1
check_command curl || exit 1
check_command bash || exit 1

# =============================================================================
# Create Directories
# =============================================================================

print_step "Creating directories..."
mkdir -p "$DATAROOT"
mkdir -p "$MODELROOT"
echo "  Created: $DATAROOT"
echo "  Created: $MODELROOT"

# =============================================================================
# Conda Environment Setup
# =============================================================================

if [[ "$CREATE_VENV" == "true" ]]; then
    print_header "Setting up Conda Environment"
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo "Error: conda is not installed or not in PATH"
        echo "Please install Miniconda or Anaconda first:"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # Initialize conda for this shell if needed
    eval "$(conda shell.bash hook)"
    
    # Check if environment already exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_step "Conda environment '$CONDA_ENV_NAME' already exists"
        read -p "Do you want to recreate it? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "Removing existing environment..."
            conda env remove -n "$CONDA_ENV_NAME" -y
            print_step "Creating new conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
            conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
        fi
    else
        print_step "Creating conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
        conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    fi
    
    print_step "Activating conda environment..."
    conda activate "$CONDA_ENV_NAME"
    echo "Python: $(which python)"
    echo "Pip: $(which pip)"
fi

# =============================================================================
# Install Dependencies
# =============================================================================

if [[ "$INSTALL_DEPS" == "true" ]]; then
    print_header "Installing Dependencies"
    
    print_step "Upgrading pip..."
    pip install --upgrade pip
    
    print_step "Installing PyTorch with ROCm support ($ROCM_VERSION_DISPLAY)..."
    echo "Using wheel URL: $PYTORCH_ROCM_URL"
    pip install $PYTORCH_PRE_FLAG torch torchvision --index-url "$PYTORCH_ROCM_URL"
    
    print_step "Installing torchtitan and requirements..."
    pip install -e .
    
    print_step "Installing base requirements..."
    pip install -r requirements.txt
    
    print_step "Installing MLPerf requirements..."
    pip install -r requirements-mlperf.txt
    
    print_step "Installing Flux requirements..."
    pip install -r torchtitan/experiments/flux/requirements-flux.txt
    
    echo ""
    echo "Dependencies installed successfully!"
fi

# =============================================================================
# Download Preprocessed Data (COCO + empty encodings)
# =============================================================================

if [[ "$DOWNLOAD_PREPROCESSED" == "true" ]]; then
    print_header "Downloading Preprocessed Validation Data"
    echo "Target directory: $DATAROOT"
    echo ""
    
    cd "$DATAROOT"
    
    print_step "Downloading preprocessed COCO validation dataset..."
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
        https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
    
    print_step "Downloading empty encodings..."
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
        https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri
    
    cd - > /dev/null
    
    echo ""
    echo "Preprocessed validation data downloaded successfully!"
fi

# =============================================================================
# Download CC12M Training Data (Optional - ~2.5TB)
# =============================================================================

if [[ "$DOWNLOAD_CC12M" == "true" ]]; then
    print_header "Downloading CC12M Training Dataset"
    echo "WARNING: This will download approximately 2.5TB of data!"
    echo "Target directory: $DATAROOT"
    echo ""
    
    cd "$DATAROOT"
    
    print_step "Downloading preprocessed CC12M dataset (~2.5TB)..."
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
        https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
    
    cd - > /dev/null
    
    echo ""
    echo "CC12M training data downloaded successfully!"
fi

# =============================================================================
# Download Raw Data (Optional)
# =============================================================================

if [[ "$DOWNLOAD_RAW" == "true" ]]; then
    print_header "Downloading Raw Datasets"
    echo "Target directory: $DATAROOT"
    echo ""
    
    cd "$DATAROOT"
    
    print_step "Downloading raw CC12M dataset..."
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
        https://training.mlcommons-storage.org/metadata/flux-1-cc12m-disk.uri
    
    print_step "Downloading raw COCO dataset..."
    bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
        https://training.mlcommons-storage.org/metadata/flux-1-coco.uri
    
    print_step "Downloading validation TSV..."
    wget -q https://training.mlcommons-storage.org/flux_1/datasets/val2014_30k.tsv
    
    cd - > /dev/null
    
    echo ""
    echo "Raw data downloaded successfully!"
fi

# =============================================================================
# Download Model Encoders (Optional)
# =============================================================================

if [[ "$DOWNLOAD_ENCODERS" == "true" ]]; then
    print_header "Downloading Model Encoders"
    echo "Target directory: $MODELROOT"
    echo ""
    
    if [[ -z "$HF_TOKEN" ]]; then
        echo "Error: HuggingFace token required for encoder download."
        echo "Use --hf-token <token> or set HF_TOKEN environment variable."
    else
        print_step "Downloading T5, CLIP, and Autoencoder..."
        python torchtitan/experiments/flux/scripts/download_encoders.py \
            --local_dir "$MODELROOT" \
            --hf_token "$HF_TOKEN"
        
        echo ""
        echo "Encoders downloaded successfully!"
    fi
fi

# =============================================================================
# Generate Environment File
# =============================================================================

print_header "Generating Environment Configuration"

ENV_FILE="env_baremetal.sh"

cat > "$ENV_FILE" << EOF
#!/bin/bash
# Flux Baremetal Training Environment
# Generated by setup_baremetal.sh on $(date)
# ROCm version: $ROCM_VERSION_DISPLAY
# Source this file before training: source $ENV_FILE

# Data and model paths
export DATAROOT="$DATAROOT"
export MODELROOT="$MODELROOT"
export LOGDIR="\${LOGDIR:-./logs}"

# Conda environment
export CONDA_ENV_NAME="$CONDA_ENV_NAME"

# HuggingFace cache
export HF_HOME="\${HF_CACHE:-\$HOME/.cache}"
export HF_HUB_CACHE="\$HF_HOME/huggingface/hub"

# ROCm/HIP settings
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export HIP_LAUNCH_BLOCKING=0
export HSA_FORCE_FINE_GRAIN_PCIE=1

# RCCL settings (compatible with NCCL)
export NCCL_DEBUG=WARN
export NCCL_BUFFSIZE=2097152

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "\$(conda shell.bash hook)"
    conda activate "\$CONDA_ENV_NAME" 2>/dev/null || echo "Warning: Could not activate conda env '\$CONDA_ENV_NAME'"
fi

echo "Flux baremetal environment loaded."
echo "  DATAROOT:  \$DATAROOT"
echo "  MODELROOT: \$MODELROOT"
echo "  LOGDIR:    \$LOGDIR"
echo "  Conda env: \$CONDA_ENV_NAME"
echo "  ROCm:      $ROCM_VERSION_DISPLAY"
EOF

chmod +x "$ENV_FILE"
echo "Environment file created: $ENV_FILE"
echo "Source it before training: source $ENV_FILE"

# =============================================================================
# Summary
# =============================================================================

print_header "Setup Complete!"

echo ""
echo "Next Steps:"
echo ""
echo "1. Source the environment file:"
echo "   source $ENV_FILE"
echo ""

if [[ "$DOWNLOAD_CC12M" == "true" ]]; then
    echo "2. Run training with preprocessed data:"
    echo "   CONFIG_FILE=./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf_preprocessed.toml \\"
    echo "   NGPU=8 bash torchtitan/experiments/flux/run_train.sh \\"
    echo "     --training.dataset_path=\$DATAROOT/cc12m_preprocessed \\"
    echo "     --eval.dataset_path=\$DATAROOT/coco_preprocessed \\"
    echo "     --encoder.empty_encodings_path=\$DATAROOT/empty_encodings"
    echo ""
elif [[ "$DOWNLOAD_PREPROCESSED" == "true" ]]; then
    echo "2. Download CC12M training data (not downloaded by default due to size):"
    echo "   cd \$DATAROOT"
    echo "   bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \\"
    echo "     https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri"
    echo ""
    echo "   Or re-run setup with --download-cc12m flag"
    echo ""
fi

if [[ "$DOWNLOAD_RAW" == "true" && "$DOWNLOAD_ENCODERS" == "true" ]]; then
    echo "   Or run training with raw data (requires encoders):"
    echo "   CONFIG_FILE=./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf.toml \\"
    echo "   NGPU=8 bash torchtitan/experiments/flux/run_train.sh \\"
    echo "     --training.dataset_path=\$DATAROOT/cc12m_disk \\"
    echo "     --eval.dataset_path=\$DATAROOT/coco \\"
    echo "     --encoder.t5_encoder=\$MODELROOT/t5 \\"
    echo "     --encoder.clip_encoder=\$MODELROOT/clip \\"
    echo "     --encoder.autoencoder_path=\$MODELROOT/autoencoder/ae.safetensors"
    echo ""
fi

echo "3. For multi-node Slurm training:"
echo "   export LOGDIR=/path/to/logs"
echo "   sbatch -N <nodes> -t <time> run_baremetal.sub"
echo ""

echo "Disk usage:"
if [[ -d "$DATAROOT" ]]; then
    du -sh "$DATAROOT" 2>/dev/null || echo "  $DATAROOT: (calculating...)"
fi
if [[ -d "$MODELROOT" ]]; then
    du -sh "$MODELROOT" 2>/dev/null || echo "  $MODELROOT: (calculating...)"
fi

echo ""
echo "For more information, see README.md"

