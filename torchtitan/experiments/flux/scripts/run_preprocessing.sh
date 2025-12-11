#!/usr/bin/bash
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Preprocessing script for Flux training data
#
# Required environment variables for baremetal:
#   DATAROOT - Path to datasets directory (e.g., /path/to/datasets)
#
# Optional environment variables:
#   NGPU     - Number of GPUs (default: 8)
#   HF_CACHE - HuggingFace cache directory (default: $HOME/.cache)
#
# Example usage:
#   DATAROOT=/path/to/data NGPU=8 ./torchtitan/experiments/flux/scripts/run_preprocessing.sh \
#     --training.dataset_path=$DATAROOT/cc12m_disk \
#     --preprocessing.output_dataset_path=$DATAROOT/cc12m_preprocessed

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf.toml"}

# HuggingFace cache (baremetal-friendly default)
export HF_HOME="${HF_CACHE:-$HOME/.cache}"
export HF_HUB_CACHE="${HF_HOME}/huggingface/hub"

# Dataset root directory
DATAROOT=${DATAROOT:-"/dataset"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# ROCm/HIP memory allocator configuration (AMD GPUs)
# For NVIDIA GPUs, change to PYTORCH_CUDA_ALLOC_CONF
PYTORCH_HIP_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.experiments.flux.scripts.preprocess_flux_dataset --job.config_file ${CONFIG_FILE} \
--experimental.custom_args_module torchtitan.experiments.flux.preprocessing_config \
--eval.dataset= \
--checkpoint.no_enable_checkpoint --training.batch_size 256 --training.dataset=cc12m_disk --training.dataset_path=${DATAROOT}/cc12m_disk \
--parallelism.data_parallel_replicate_degree=${NGPU} \
--training.classifer_free_guidance_prob=0.0 \
--model.flavor=flux-debug $overrides

mkdir -p ${DATAROOT}/empty_encodings
PYTORCH_HIP_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
-m torchtitan.experiments.flux.scripts.save_empty_encodings --job.config_file ${CONFIG_FILE} \
--experimental.custom_args_module torchtitan.experiments.flux.preprocessing_config \
--eval.dataset= \
--preprocessing.output_dataset_path=${DATAROOT}/empty_encodings \
--training.classifer_free_guidance_prob=0.0 \
--checkpoint.no_enable_checkpoint --training.batch_size 256 --training.dataset=dummy \
--model.flavor=flux-debug --encoder.empty_encodings_path=
