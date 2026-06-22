import json
import pandas as pd
import torch
from pathlib import Path
from PIL import Image

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)

from peft import PeftModel

from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore


# ======================
# CONFIG
# ======================

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
LORA_PATH = "internvl2_5_finetune/models_english"

TEST_JSON = "URDU/merge_test_english.json"
IMAGE_DIR = "internvl2_5_finetune"

MAX_NEW_TOKENS = 64
DEVICE = "cuda"

BATCH_SIZE = 12   # 👈 change based on GPU (3060 → 1–2 recommended)


# ======================
# LOAD TEST DATA
# ======================

with open(TEST_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

image_map = {
    x["id"]: x["file_name"]
    for x in coco["images"]
}

annotations = coco["annotations"]

print("Test samples:", len(annotations))


# ======================
# LOAD MODEL
# ======================

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()


# ======================
# STORAGE
# ======================

references = []
predictions = []
results = []


# ======================
# BATCH INFERENCE LOOP
# ======================

for i in range(0, len(annotations), BATCH_SIZE):

    batch_rows = annotations[i:i + BATCH_SIZE]

    batch_images = []
    batch_texts = []
    batch_refs = []
    batch_questions = []

    # ----------------------
    # Prepare batch
    # ----------------------
    for row in batch_rows:

        image_path = Path(IMAGE_DIR) / image_map[row["image_id"]]
        image = Image.open(image_path).convert("RGB")

        question = row["Question"]
        gt_answer = row["Answer"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        batch_images.append(image)
        batch_texts.append(text)
        batch_refs.append(gt_answer)
        batch_questions.append(question)

    # ----------------------
    # Tokenize batch
    # ----------------------
    inputs = processor(
        text=batch_texts,
        images=batch_images,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    # ----------------------
    # Generate
    # ----------------------
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

    # ----------------------
    # Remove prompt tokens
    # ----------------------
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, prompt_len:]

    # ----------------------
    # Decode
    # ----------------------
    preds = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )

    preds = [p.strip() for p in preds]

    # ----------------------
    # Save results
    # ----------------------
    for q, ref, pred in zip(batch_questions, batch_refs, preds):

        references.append(ref)
        predictions.append(pred)

        results.append({
            "question": q,
            "reference": ref,
            "prediction": pred
        })

    if i % (BATCH_SIZE * 25) == 0:
        print(f"{i}/{len(annotations)}")


# ======================
# SAVE OUTPUT
# ======================

df = pd.DataFrame(results)
df.to_csv("predictions__english_only_english.csv", index=False)

print("Saved predictions_wbc_english.csv")


# ======================
# BLEU SCORE
# ======================

bleu_refs = [[x.split()] for x in references]
bleu_preds = [x.split() for x in predictions]

bleu = corpus_bleu(bleu_refs, bleu_preds)

print("\nBLEU:", bleu)


# ======================
# ROUGE SCORE
# ======================

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

r1, r2, rl = [], [], []

for pred, ref in zip(predictions, references):

    scores = scorer.score(ref, pred)

    r1.append(scores["rouge1"].fmeasure)
    r2.append(scores["rouge2"].fmeasure)
    rl.append(scores["rougeL"].fmeasure)

print("\nROUGE-1:", sum(r1) / len(r1))
print("ROUGE-2:", sum(r2) / len(r2))
print("ROUGE-L:", sum(rl) / len(rl))


# ======================
# BERT SCORE
# ======================
from bert_score import score as bertscore

# make sure both are lists of strings
predictions = [p.strip() for p in predictions if len(p.strip()) > 0]
references  = [r.strip() for r in references if len(r.strip()) > 0]

P, R, F1 = bertscore(
    predictions,
    references,
    lang="en",
    model_type="roberta-large",
    batch_size=8,
    verbose=True
)

print("\nBERTScore Precision:", P.mean().item())
print("BERTScore Recall:", R.mean().item())
print("BERTScore F1:", F1.mean().item())
