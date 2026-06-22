import json
import torch
import importlib
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

CONFIG = importlib.import_module("00_config").CONFIG


# =========================
# DATASET
# =========================
class EnglishVQADataset(Dataset):

    def __init__(self, json_path, image_dir, processor):

        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.image_dir = Path(image_dir)
        self.processor = processor

        # map image_id -> file_name
        self.image_map = {
            img["id"]: img["file_name"]
            for img in coco["images"]
        }

        self.data = coco["annotations"]

    def __len__(self):
        return len(self.data)

    def _resolve_image_path(self, image_field):

        image_path = Path(image_field)

        if not image_path.is_absolute():
            image_path = self.image_dir / image_path

        return image_path

    def __getitem__(self, idx):

        row = self.data[idx]
        image_id = row["image_id"]
        image_file = self.image_map[image_id]
        image_path = self._resolve_image_path(image_file)

        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        # print(row)
        question = row["Question"]
        answer = row["Answer"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        labels = input_ids.clone()

        # mask everything first
        labels[:] = -100

        # only train last part (simple fallback)
        labels[-len(answer):] = input_ids[-len(answer):]

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }

        if "mm_token_type_ids" in inputs:
            output["mm_token_type_ids"] = inputs["mm_token_type_ids"][0]

        return output


# =========================
# COLLATOR
# =========================
def data_collator(features):

    batch = {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
        "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
        "image_grid_thw": torch.cat([f["image_grid_thw"] for f in features], dim=0),
    }

    if "mm_token_type_ids" in features[0]:
        batch["mm_token_type_ids"] = torch.stack(
            [f["mm_token_type_ids"] for f in features]
        )

    return batch


# =========================
# CHECKPOINT
# =========================
def get_latest_checkpoint(output_dir):

    output_dir = Path(output_dir)

    checkpoints = list(output_dir.glob("checkpoint-*"))

    if not checkpoints:
        return None

    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(x.name.split("-")[-1])
    )

    return str(checkpoints[-1])


# =========================
# MAIN
# =========================
def main():

    data_dir = Path(CONFIG["data_dir"])

    train_json = data_dir / "URDU//WBC_leukemia_ATT_VQA_validate_hajra.json"
    test_json = data_dir / "URDU/Q_A_test_large_validate.json"

    train_image_dir = Path(CONFIG["train_cropped_dir"])
    test_image_dir = Path(CONFIG["test_cropped_dir"])

    output_dir = Path(CONFIG["model_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        CONFIG["model_id"],
        trust_remote_code=True,
    )
    print("Processor loaded.")

    print("\nLoading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        CONFIG["model_id"],
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2",
        device_map={"": 0},
        trust_remote_code=True,
        
    )
    print("Model loaded.")
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print("\nApplying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading datasets...")

    train_dataset = EnglishVQADataset(
        train_json,
        train_image_dir,
        processor,
    )

    test_dataset = EnglishVQADataset(
        test_json,
        test_image_dir,
        processor,
    )

    print(f"Train rows: {len(train_dataset)}")
    print(f"Test rows:  {len(test_dataset)}")
# 
    training_args = TrainingArguments(

        output_dir=str(output_dir),

        num_train_epochs=CONFIG["num_train_epochs"],

        per_device_train_batch_size=CONFIG["batch_size"],

        per_device_eval_batch_size=CONFIG["batch_size"],  # ✅ ADDED

        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],

        learning_rate=CONFIG["learning_rate"],

        logging_steps=CONFIG["logging_steps"],

        save_steps=CONFIG["save_steps"],
        save_strategy="steps",
        save_total_limit=2,

        # evaluation_strategy="steps",   # ✅ ADDED
        eval_steps=CONFIG["save_steps"],

        remove_unused_columns=False,

        dataloader_num_workers=0,

        gradient_checkpointing=True,

        max_grad_norm=1.0,

        report_to="none",

        fp16=False,
        bf16=True,
    )

    trainer = Trainer(

        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,   # ✅ ADDED
        data_collator=data_collator,
    )

    latest_checkpoint = get_latest_checkpoint(output_dir)

    if latest_checkpoint:
        print(f"\nResuming training from: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("\nStarting fresh training...")
        trainer.train()

    print("\nSaving final model...")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    print("\nTRAINING COMPLETE")
    print(f"Saved at: {output_dir}")


if __name__ == "__main__":
    main()