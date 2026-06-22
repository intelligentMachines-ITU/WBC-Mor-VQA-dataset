
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
LORA_PATH = "internvl2_5_finetune/models"

TEST_JSON = "QWEN/Q_A_test_large_validate.json"
IMAGE_DIR =  "internvl2_5_finetune"

MAX_NEW_TOKENS = 64
DEVICE = "cuda"


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
# GENERATE
# ======================

references = []
predictions = []

results = []

for i,row in enumerate(annotations):

    image_path = Path(
        IMAGE_DIR
    ) / image_map[row["image_id"]]

    image = Image.open(
        image_path
    ).convert("RGB")

    question = row["Question_urdu"]
    gt_answer = row["Answer_urdu"]

    messages = [

        {
            "role":"user",
            "content":[
                {
                    "type":"image",
                    "image":image
                },
                {
                    "type":"text",
                    "text":question
                }
            ]
        }

    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )

    inputs = {
        k:v.to(model.device)
        for k,v in inputs.items()
    }

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

    generated_ids = generated_ids[
        :,inputs["input_ids"].shape[1]:
    ]

    pred = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    pred = pred.strip()

    predictions.append(pred)
    references.append(gt_answer)

    results.append({

        "question":question,
        "reference":gt_answer,
        "prediction":pred

    })

    if i%100==0:

        print(
            f"{i}/{len(annotations)}"
        )


# ======================
# SAVE PREDICTIONS
# ======================

df = pd.DataFrame(results)

df.to_csv(
    "predictions_leu.csv",
    index=False
)

print("Saved predictions.csv")


# ======================
# BLEU
# ======================

bleu_refs = [
    [x.split()]
    for x in references
]

bleu_preds = [
    x.split()
    for x in predictions
]

bleu = corpus_bleu(
    bleu_refs,
    bleu_preds
)

print("\nBLEU:",bleu)


# ======================
# ROUGE
# ======================

scorer = rouge_scorer.RougeScorer(
    ['rouge1','rouge2','rougeL'],
    use_stemmer=True
)

r1=[]
r2=[]
rl=[]

for pred,ref in zip(
    predictions,
    references
):

    scores = scorer.score(
        ref,
        pred
    )

    r1.append(
        scores["rouge1"].fmeasure
    )

    r2.append(
        scores["rouge2"].fmeasure
    )

    rl.append(
        scores["rougeL"].fmeasure
    )


print(
    "\nROUGE-1:",
    sum(r1)/len(r1)
)

print(
    "ROUGE-2:",
    sum(r2)/len(r2)
)

print(
    "ROUGE-L:",
    sum(rl)/len(rl)
)


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
