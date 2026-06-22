# WBCMor-VQA Training and Fine-Tuning Guide

## Overview

This repository provides the resources required to train, fine-tune, and evaluate vision-language models on the WBCMor-VQA dataset.

The guide covers:

* Environment setup
* CUDA installation
* PyTorch installation
* Dataset preparation
* Fine-tuning
* Inference
* Evaluation
* Citation

---

# System Requirements

## Recommended Hardware

### Minimum

* NVIDIA GPU with 12 GB VRAM
* 32 GB RAM
* 100 GB available storage

### Recommended

* NVIDIA GPU with 16 GB+ VRAM
* 64 GB RAM
* CUDA-compatible GPU
* Linux or Windows

---

# Environment Setup

## Step 1: Install Anaconda / Miniconda

Download and install Anaconda or Miniconda.

Verify installation:

```bash
conda --version
```

---

## Step 2: Create Environment

Create a new environment:

```bash
conda create -n wbcmor-vqa python=3.10 -y
```

Activate environment:

```bash
conda activate wbcmor-vqa
```

Verify Python:

```bash
python --version
```

---

# CUDA Installation

## Check GPU

Verify that an NVIDIA GPU is available:

```bash
nvidia-smi
```

If the command runs successfully, your GPU drivers are installed correctly.

---

## Install CUDA

Download CUDA Toolkit from NVIDIA.

Recommended version:

```text
CUDA 12.x
```

After installation verify:

```bash
nvcc --version
```

---

# PyTorch Installation

Install PyTorch compatible with your CUDA version.

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

---

# Install Required Packages

Install all dependencies:

```bash
pip install transformers
pip install accelerate
pip install peft
pip install bitsandbytes
pip install datasets
pip install evaluate
pip install rouge-score
pip install bert-score
pip install nltk
pip install pandas
pip install tqdm
pip install pillow
pip install openpyxl
pip install qwen-vl-utils
```

Alternatively:

```bash
pip install -r requirements.txt
```

---

# Repository Structure

```text
WBCMor-VQA/

├── data/
│   ├── train/
│   ├── test_english/
│   └── test_urdu/
│
├── training/
│   ├── qwen2vl/
│   ├── internvl2.5/
│   └── unihema/
│
├── images/
│
├── docs/
│
└── README.md
```

---

# Dataset Preparation

Place training annotations inside:

```text
data/train/
```

Place English test annotations inside:

```text
data/test_english/
```

Place Urdu test annotations inside:

```text
data/test_urdu/
```

Place image files inside:

```text
images/
```

Ensure that image paths referenced in the JSON files match the image directory structure.

---

# Fine-Tuning

## Qwen2-VL

Navigate to:

```bash
cd training/qwen2vl
```

Run training:

```bash
python train.py
```

---

## InternVL2.5

Navigate to:

```bash
cd training/internvl2.5
```

Run training:

```bash
python train.py
```

---

## Uni-Hema

Navigate to:

```bash
cd training/unihema
```

Run training:

```bash
python train.py
```

---

# Inference

After training completes, generate predictions using:

```bash
python inference.py
```

Predictions will be saved to the output directory specified in the configuration file.

---

# Configuration

Training parameters can be modified in the configuration file provided with each model.

Common parameters include:

* Model checkpoint
* Dataset path
* Batch size
* Learning rate
* Number of epochs
* Output directory

Adjust these settings according to available hardware resources.

---

# Troubleshooting

## CUDA Not Detected

Check:

```bash
nvidia-smi
```

and

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Out of Memory Error

Reduce:

* Batch size
* Image resolution
* Gradient accumulation steps

---

## Package Errors

Update packages:

```bash
pip install --upgrade transformers accelerate peft
```

---

## License

This repository and dataset are distributed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

You are free to share, adapt, and redistribute the material for any purpose, provided appropriate credit is given to the original authors.

For complete license details, please see the `LICENSE` file or visit:

https://creativecommons.org/licenses/by/4.0/

# Citation

If you use WBCMor-VQA in your research, please cite:

```bibtex
@article{WBCMorVQA,
  title={WBCMor-VQA: A Multilingual Hematopathology Visual Question Answering Dataset},
  author={Authors},
  year={2026}
}
```

Citation details will be updated upon publication.

---
