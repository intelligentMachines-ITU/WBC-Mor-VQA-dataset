# Multilingual Hematology Visual Question Answering Dataset

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
git clone (https://github.com/intelligentMachines-ITU/WBC-Mor-VQA-dataset)
cd WBC-Mor-VQA-dataset
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
The dataset was developed using morphology annotations derived from:

* LeukemiaAttri (MICCAI 2024)
* WBCAtt-VQA (Uni-Hema, CVPR 2026)
* 
The WBCMor-VQA dataset can be downloaded from:

📂 https://figshare.com/articles/dataset/WBC-Mor-VQA_Multilingual_Hematology_Visual_Question_Answering_dataset/32727159


## Training
### Step 01
Before training, update the dataset paths and training parameters in:

```bash
00_config.py
```

### Step 02 Start training:

```bash
python 01_train.py
```

---

## Testing

Evaluate the trained model:

```bash
python 02_test.py
```

---

## Citation

If you use WBCMor-VQA in your research, please also consider citing the source datasets used in its construction.

**Citation information for WBCMor-VQA will be added upon acceptance/publication of the associated paper.**

### WBCAtt

```bibtex
@article{tsutsui2023wbcatt,
  title={Wbcatt: A White Blood Cell Dataset Annotated with Detailed Morphological Attributes},
  author={Tsutsui, Satoshi and Pang, Winnie and Wen, Bihan},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={50796--50824},
  year={2023}
}
```

### LeukemiaAttri

```bibtex
@inproceedings{rehman2024leukemiaattri,
  title={A Large-Scale Multi Domain Leukemia Dataset for the White Blood Cells Detection with Morphological Attributes for Explainability},
  author={Rehman, Abdul and Meraj, Talha and Minhas, Aiman Mahmood and Imran, Ayisha and Ali, Mohsen and Sultani, Waqas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={553--563},
  year={2024},
  organization={Springer}
}
```


### UniHema

```bibtex
@inproceedings{rehman2026uni,
  title={Uni-Hema: Unified Model for Digital Hematopathology},
  author={Rehman, Abdul and Rasool, Iqra and Imran, Ayisha and Ali, Mohsen and Sultani, Waqas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={37578--37589},
  year={2026}
}
```


