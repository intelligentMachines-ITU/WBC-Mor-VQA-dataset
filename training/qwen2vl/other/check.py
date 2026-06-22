import json
from collections import defaultdict

json_file = "/home/administrator/Abdul_Rehman/QWEN/WBC_leukemia_ATT_VQA.json" 

with open(json_file, "r") as f:
    coco = json.load(f)

# Count annotations per image
ann_count = defaultdict(int)

for ann in coco["annotations"]:
    ann_count[ann["image_id"]] += 1

missing_annotations = []

for img in coco["images"]:
    image_id = img["id"]

    if ann_count[image_id] == 0:
        missing_annotations.append(img)

print(f"Total images: {len(coco['images'])}")
print(f"Images without annotations: {len(missing_annotations)}")

if missing_annotations:
    print("\nExamples:")
    for img in missing_annotations[:10]:
        print(img["id"], img["file_name"])


import json
from pathlib import Path

# json_file = "your_coco.json"
image_root = Path("/home/administrator/Abdul_Rehman/QWEN/")  # change this

with open(json_file, "r") as f:
    coco = json.load(f)

missing_files = []

for img in coco["images"]:
    file_name = img["file_name"]
    img_path = image_root / file_name

    if not img_path.exists():
        missing_files.append((img["id"], file_name))

print(f"Total images: {len(coco['images'])}")
print(f"Missing files: {len(missing_files)}")

if missing_files:
    print("\nExamples of missing images:")
    for x in missing_files[:10]:
        print("image_id:", x[0], "file_name:", x[1])