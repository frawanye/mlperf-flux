#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# =============================================================================
# Baremetal Flux Training Script for AMD GPUs (ROCm)
# =============================================================================
#
# Prerequisites:
#   source env_baremetal.sh   # Sets DATAROOT, activates conda env
#
# Usage:
#   NGPU=4 ./torchtitan/experiments/flux/run_train.sh [additional args...]
#
# Environment variables (set by env_baremetal.sh):
#   DATAROOT    - Path to datasets directory (REQUIRED)
#   MODELROOT   - Path to model encoders (optional, for non-preprocessed data)
#   NGPU        - Number of GPUs (default: 8)
#   CONFIG_FILE - Config file path (default: preprocessed config)
#
# The Python code will automatically find datasets in $DATAROOT/<dataset_name>

# =============================================================================
# Validate environment
# =============================================================================
if [ -z "${DATAROOT}" ]; then
    echo "ERROR: DATAROOT environment variable is not set."
    echo ""
    echo "Please source env_baremetal.sh first:"
    echo "  source env_baremetal.sh"
    echo ""
    echo "Or set DATAROOT manually:"
    echo "  export DATAROOT=/path/to/datasets"
    exit 1
fi

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf_preprocessed.toml"}
DEBUG=${DEBUG:-0}

# Ensure DATAROOT is exported for Python to access
export DATAROOT
export MODELROOT

# HuggingFace cache
export HF_HOME="${HF_CACHE:-$HOME/.cache}"
export HF_HUB_CACHE="${HF_HOME}/huggingface/hub"

if [ $DEBUG == 1 ]; then
    DEBUG_FLAG="-m debugpy --listen 0.0.0.0:5678 --wait-for-client"
else
    DEBUG_FLAG=""
fi

# Collect user overrides
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

echo "Training configuration:"
echo "  DATAROOT:    ${DATAROOT}"
echo "  MODELROOT:   ${MODELROOT:-"(not set)"}"
echo "  NGPU:        ${NGPU}"
echo "  CONFIG_FILE: ${CONFIG_FILE}"

# =============================================================================
# Launch training
# =============================================================================
# ROCm/HIP memory allocator configuration (AMD GPUs)
# For NVIDIA GPUs, change to PYTORCH_CUDA_ALLOC_CONF
PYTORCH_HIP_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="127.0.0.1:29500" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
${DEBUG_FLAG} \
-m torchtitan.experiments.flux.train --job.config_file ${CONFIG_FILE} $overrides
