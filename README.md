# 1. Problem 

## Text to Image - Flux.1-schnell.

[Torchtitan](https://github.com/pytorch/torchtitan) provides an implementation of the Flux model from [Black Forest Labs](https://bfl.ai/). We adapt this for MLPerf Training. The relevant files are under `torchtitan/experiments/flux`.
These files plug in to the rest of torchtitan.

```
@inproceedings{
   liang2025torchtitan,
   title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
   author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```

## Branch Modifications (vs. upstream torchtitan)

This branch adds support for **baremetal execution on AMD GPUs with ROCm**. Key changes from upstream:

| Category | Modification |
|----------|-------------|
| **FSDP2 Compatibility** | Added `torchtitan/distributed/fsdp_compat.py` to support ROCm PyTorch builds where FSDP2 APIs are under `torch.distributed._composable.fsdp` |
| **Path Resolution** | Configs use empty defaults; paths are resolved at runtime via `DATAROOT`/`MODELROOT` environment variables |
| **ROCm Environment** | Replaced CUDA-specific env vars (`PYTORCH_CUDA_ALLOC_CONF`) with ROCm equivalents (`PYTORCH_HIP_ALLOC_CONF`) |
| **Setup Automation** | Added `setup_baremetal.sh` for automated conda env setup, PyTorch+ROCm installation, and data downloads |
| **Slurm Support** | Added `run_baremetal.sub` for multi-node training on HPC clusters |
| **Config Files** | Added/modified config files for different training scenarios.

---

# 2. Directions

## Quick Start

The fastest way to get started:

```bash
# 1. Run the setup script (creates conda env, installs dependencies)
./setup_baremetal.sh --download-cc12m --dataroot /path/to/data --modelroot /path/to/models

# 3. Source the generated environment file
source env_baremetal.sh

# 4. Run training
NGPU=8 ./torchtitan/experiments/flux/run_train.sh --training.batch_size=16 --training.seed=1234
```

## Detailed Setup Options

### Option A: Automated Setup (Recommended)

The `setup_baremetal.sh` script handles environment setup, dependency installation, and data downloads:

```bash
# Basic setup - downloads validation data only (COCO + empty_encodings)
./setup_baremetal.sh --dataroot /path/to/data --modelroot /path/to/models

# Include CC12M training data (~2.5TB)
./setup_baremetal.sh --download-cc12m --dataroot /path/to/data --modelroot /path/to/models

# Use specific ROCm version (auto-detects by default)
./setup_baremetal.sh --rocm-version 6.2 --dataroot /path/to/data

# Full setup including raw data and encoders
./setup_baremetal.sh --all --hf-token <your_token> --dataroot /path/to/data
```

The script:
- Creates a conda environment (`flux-mlperf` by default)
- Auto-detects your ROCm version and installs matching PyTorch
- Installs all required dependencies
- Downloads preprocessed validation data
- Generates `env_baremetal.sh` with all required environment variables

Run `./setup_baremetal.sh --help` for all options.

### Option B: Manual Setup

#### 1. Install ROCm and PyTorch

Ensure ROCm is installed on your system. Then install PyTorch with ROCm support:

```bash
# For ROCm 6.4 (latest stable)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

# For ROCm 6.2.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# For ROCm 7.1 (nightly)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1
```

#### 2. Install Dependencies

```bash
pip install -e .
pip install -r requirements.txt
pip install -r requirements-mlperf.txt
pip install -r torchtitan/experiments/flux/requirements-flux.txt
```

#### 3. Set Environment Variables

Create an environment file or export directly:

```bash
export DATAROOT=/path/to/datasets
export MODELROOT=/path/to/models  # Only needed for non-preprocessed data
export LOGDIR=/path/to/logs

# ROCm-specific settings
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export HIP_LAUNCH_BLOCKING=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
```

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DATAROOT` | Yes | Path to datasets directory |
| `MODELROOT` | For raw data | Path to model encoders (T5, CLIP, autoencoder) |
| `LOGDIR` | For Slurm | Path to log output directory |
| `NGPU` | No | Number of GPUs per node (default: 8) |
| `CONFIG_FILE` | No | Path to training config TOML |
| `SEED` | No | Random seed (default: 1234) |
| `HF_CACHE` | No | HuggingFace cache directory (default: `$HOME/.cache`) |
| `CONDA_ENV_NAME` | No | Conda environment name (default: `flux-mlperf`) |

### ROCm-Specific Variables (set automatically by env_baremetal.sh)

| Variable | Description |
|----------|-------------|
| `PYTORCH_HIP_ALLOC_CONF` | HIP memory allocator configuration (not supported by the GPUs I tested on)|
| `HIP_LAUNCH_BLOCKING` | HIP debugging flag (0=async, 1=sync) |
| `HSA_FORCE_FINE_GRAIN_PCIE` | HSA memory settings for AMD GPUs |
| `NCCL_*` | RCCL settings (compatible with NCCL variable names) |

---

## Steps to Download and Verify Data

### Preprocessed Data (Recommended)

Training on preprocessed embeddings avoids loading encoders during training and is faster:

```bash
cd $DATAROOT

# Preprocessed COCO validation (required)
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri

# Empty encodings for guidance (required)
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri

# Preprocessed CC12M training data (~2.5TB)
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
```

### Raw Data (Alternative)

If you prefer to run preprocessing yourself or train without preprocessing:

```bash
cd $DATAROOT

# Raw CC12M dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-cc12m-disk.uri

# Raw COCO validation dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://training.mlcommons-storage.org/metadata/flux-1-coco.uri
wget https://training.mlcommons-storage.org/flux_1/datasets/val2014_30k.tsv
```

### Download Model Encoders (for raw data only)

Required only if training from raw (non-preprocessed) data:

```bash
python torchtitan/experiments/flux/scripts/download_encoders.py \
  --local_dir $MODELROOT \
  --hf_token <your_access_token>
```

You need access rights to https://huggingface.co/black-forest-labs/FLUX.1-schnell.

---

## Running Training

### Single-Node Training

```bash
# Source environment (sets DATAROOT, activates conda, etc.)
source env_baremetal.sh

# Run with preprocessed data
NGPU=4 ./torchtitan/experiments/flux/run_train.sh \
  --training.batch_size=16 \
  --training.seed=1234

# Run with custom config
CONFIG_FILE=./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf.toml \
NGPU=8 ./torchtitan/experiments/flux/run_train.sh \
  --training.batch_size=16
```

### Multi-Node Slurm Training

```bash
# Source environment first
source env_baremetal.sh

# Set log directory
export LOGDIR=/path/to/logs

# Submit job (GPU resources passed via sbatch)
sbatch --nodes=4 --gpus-per-node=4 --time=04:00:00 run_baremetal.sub

# Or with additional training overrides
sbatch --nodes=4 --gpus-per-node=4 run_baremetal.sub --training.batch_size=32
```

**Note:** Edit `run_baremetal.sub` to customize module loading for your cluster. GPU resources should be requested via `sbatch` command-line options rather than hardcoded in the script.

---

## Preprocessing (Optional)

Since the encoders are frozen during training, you can preprocess data offline to avoid encoding on the fly.

```bash
export DATAROOT=/path/to/datasets

# Preprocess CC12M training data
NGPU=8 torchtitan/experiments/flux/scripts/run_preprocessing.sh \
  --training.dataset_path=$DATAROOT/cc12m_disk \
  --training.dataset=cc12m_disk \
  --eval.dataset= \
  --training.batch_size=256 \
  --preprocessing.output_dataset_path=$DATAROOT/cc12m_preprocessed

# Preprocess COCO validation data
NGPU=4 torchtitan/experiments/flux/scripts/run_preprocessing.sh \
  --training.dataset=coco \
  --training.dataset_path=$DATAROOT/coco \
  --eval.dataset= \
  --training.batch_size=128 \
  --preprocessing.output_dataset_path=$DATAROOT/coco_preprocessed
```

**Note:** Due to dataset size, the number of samples must be divisible by `batch_size × NGPUs`.

---

## Training Configuration

The training script uses TOML config files in `torchtitan/experiments/flux/train_configs/`. Parameters can be overridden via CLI:

```bash
# See all available parameters
CONFIG_FILE=./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf_preprocessed.toml \
NGPU=1 ./torchtitan/experiments/flux/run_train.sh --help
```

### Key Config Files

| Config | Description |
|--------|-------------|
| `flux_schnell_mlperf_preprocessed.toml` | **Recommended.** For preprocessed embeddings |
| `flux_schnell_mlperf.toml` | For raw data (requires encoders) |

### Common Overrides

```bash
--training.batch_size=16          # Batch size per GPU
--training.seed=1234              # Random seed
--training.steps=1000             # Training steps
--training.compile                # Enable torch.compile
--parallelism.data_parallel_replicate_degree=N  # DDP across N nodes
--parallelism.data_parallel_shard_degree=N      # FSDP shard degree
--checkpoint.enable_checkpoint    # Enable checkpointing
--checkpoint.interval=1000        # Checkpoint every N steps
```

---

# 3. Dataset/Environment

### Publication/Attribution
We use the CC12M dataset available at https://huggingface.co/datasets/pixparse/cc12m-wds

```
@inproceedings{changpinyo2021cc12m,
  title = {{Conceptual 12M}: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts},
  author = {Changpinyo, Soravit and Sharma, Piyush and Ding, Nan and Soricut, Radu},
  booktitle = {CVPR},
  year = {2021},
}
```

We use the COCO2014 dataset for validation.

```
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={Computer vision--ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

### Data preprocessing
For both datasets, images are resized to 256x256 using bicubic interpolation.

The ~10% of the CC12M dataset is used (1,099,776 samples).
The COCO-2014-validation dataset consists of 40,504 images and 202,654 annotations. 
However, our benchmark uses only a subset of 29,696 images and annotations chosen at random with a preset seed.

Optionally, the training and validation datasets are preprocessed by running the encoders offline before training.

---

# 4. Model

### Publication/Attribution
This model largely follows the Flux.1-schnell model, as implemented by torchtitan.
The model code is largely based on the model open-sourced in [huggingface](https://huggingface.co/black-forest-labs/FLUX.1-schnell) by [Black Forest Labs](https://bfl.ai/).

```
@inproceedings{esser2024scaling,
  title={Scaling rectified flow transformers for high-resolution image synthesis},
  author={Esser, Patrick and Kulal, Sumith and Blattmann, Andreas and Entezari, Rahim and M{\"u}ller, Jonas and Saini, Harry and Levi, Yam and Lorenz, Dominik and Sauer, Axel and Boesel, Frederic and others},
  booktitle={Forty-first international conference on machine learning},
  year={2024}
}
```

### List of layers 

| **Component** | **Architecture** | **Parameters** | **Technical Details** |
|---------------|------------------|----------------|----------------------|
| **Text Encoders (Frozen)** | | | |
| └ [VIT-L CLIP text encoder](https://huggingface.co/openai/clip-vit-large-patch14) | Transformer | ~123M | Max sequence length: 77 tokens |
| | | | Output dimension: 768 |
| └ [T5-XXL](https://huggingface.co/google/t5-v1_1-xxl) | Transformer | ~11B | Max sequence length: 256 tokens |
| | |  | Output dimension: 4096 |
| **Image Encoder (Frozen)** | | | |
| └ [VAE (Variational AutoEncoder)](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | CNN | ~84M | Downscaling factor: 8 (256→32) |
| | | | Channel depth: 16 |
| **Diffusion Transformer** | | | |
| └ [Flux Diffusion Transformer](https://github.com/black-forest-labs/flux/) | Multimodal Diffusion Transformer (MMDiT) | ~11.9B |
| | **Double Stream Blocks** | | **19 layers** |
| | **Single Stream Blocks** | | **38 layers** |
| | | | 24 attention heads per layer | 
| | | | Hidden dimension: 3072 |
| | | | MLP ratio: 4.0 | | Processes 64 input channels |

### Loss function
The MSE calculated over latents is used for the loss

### Optimizer
AdamW

### Precision
The model runs with BF16 by default. This can be changed by setting `--training.mixed_precision_param=float32`.

### Weight initialization
The weight initialization strategy is taken from torchtitan. It consists of a mixture of constant, Xavier and Normal initialization.
For precise details, consult the code at `torchtitan/experiments/flux/model/model.py:init_weights`.

---

# 5. Quality

### Quality metric
Validation loss averaged over 8 equidistant time steps [0, 7/8], as described in [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206).
The validation dataset is prepared in advance so that each sample is associated with a timestep.
This is an integer from 0 to 7 inclusive, and thus should be divided by `8.0` to obtain the timestep.

The algorithm is as follows:

```pseudocode
ALGORITHM: Validation Loss Computation

INPUT:
  - validation_samples: set of validation data samples
  - num_timesteps: 8 (number of equidistant time steps)

INITIALIZE:
  - sum[8]: array of zeros for accumulating losses
  - count[8]: array of zeros for counting samples per timestep
  - t: 0 (current timestep index)

FOR each sample in validation_samples:
    loss = forward_pass(sample, timestamp=t/8)
    sum[t] += loss
    count[t] += 1
    t = (t + 1) % num_timesteps

mean_per_timestep = sum / count
validation_loss = mean(mean_per_timestep)

RETURN validation_loss
```

As we ensure that the validation set has an equal number of samples per timestep, 
a simple average of all loss values is equivalent to the above.

### Quality target
0.586

### Evaluation frequency
Every 262,144 training samples.

### Evaluation thoroughness
29,696 samples
