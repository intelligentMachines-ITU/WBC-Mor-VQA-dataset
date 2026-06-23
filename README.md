# Qwen2-VL Fine-Tuning for WBCMor-VQA

### Authors

Hajra Malik, Hafiza Tooba Aftab, Abdul Rehman, Mohsen Ali, Waqas Sultani

### Dataset

📂 Dataset: [WBCMor-VQA on Figshare](https://figshare.com/articles/dataset/WBC-Mor-VQA_Multilingual_Hematology_Visual_Question_Answering_dataset/32727159)

## Model Weights

🤗 Model Weights: [ Model Weights on Figshare](https://figshare.com/articles/dataset/WBC-Mor-VQA_Multilingual_Hematology_Visual_Question_Answering_dataset/32727159)

Available checkpoints:

- Bilingual_Weights
- Only_English_Weights

---

## Abstract

WBCMor-VQA is a bilingual (English–Urdu) hematopathology visual question answering dataset containing morphology-aware question-answer pairs associated with microscopic images of normal and leukemic white blood cells. This repository provides code and model weights for fine-tuning Qwen2-VL on WBCMor-VQA. The provided implementation supports multilingual hematopathology visual question answering and enables training, evaluation, and inference on English and Urdu datasets.

---

## Installation

We recommend the use of a Linux machine equipped with CUDA-compatible GPUs. The execution environment can be installed through Conda.

Clone repository:

```bash
git clone <repository_link>
cd <repository_name>
```

Create environment:

```bash
conda create --name wbcmor_vqa python=3.10
conda activate wbcmor_vqa
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version.

---

## Dataset

The WBCMor-VQA dataset can be downloaded from:

📂 https://figshare.com/articles/dataset/WBC-Mor-VQA_Multilingual_Hematology_Visual_Question_Answering_dataset/32727159

The dataset was developed using morphology annotations derived from:

* LeukemiaAttri (MICCAI 2024)
* WBCAtt-VQA (Uni-Hema, CVPR 2026)

Update dataset paths in:

```bash
00_config.py
```

before training.

---

## Training

Before training, update the dataset paths and training parameters in:

```bash
00_config.py
```

Start training:

```bash
python 01_train.py
```

---

## Testing

Evaluate the trained model:

```bash
python 02_test.py
```

Generate predictions:

```bash
python test.py
```

---

## Citation
### WBCAtt-VQA

```bibtex
@inproceedings{rehman2026uni,
  title={Uni-Hema: Unified Model for Digital Hematopathology},
  author={Rehman, Abdul and Rasool, Iqra and Imran, Ayisha and Ali, Mohsen and Sultani, Waqas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={37578--37589},
  year={2026}
}
```


