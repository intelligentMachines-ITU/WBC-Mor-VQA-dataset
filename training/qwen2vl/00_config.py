from pathlib import Path

BASE_DIR = Path(r"/media/iml/CVML10/Hajra/internvl2_5_finetune")

CONFIG = {
    "base_dir": BASE_DIR,

    "train_image_dir": BASE_DIR / "train",
    "test_image_dir": BASE_DIR / "test",

    "train_cropped_dir": BASE_DIR,
    "test_cropped_dir": BASE_DIR,

    "data_dir": BASE_DIR,
    "model_output_dir": BASE_DIR / "models_english",
    "results_dir": BASE_DIR / "results_english",

    "model_id": "Qwen/Qwen2-VL-2B-Instruct",

    "max_length": 512,
    "num_train_epochs": 3,
    "batch_size": 12,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "save_steps": 200,
    "logging_steps": 10,

    "max_new_tokens": 64,
    "num_workers":0,
}