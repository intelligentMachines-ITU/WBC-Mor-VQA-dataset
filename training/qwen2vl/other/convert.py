import json
import re
from pathlib import Path
from PIL import Image

# Paths
input_json = "Test_cleaned_validated_dataset.json"
image_root = "test/test_resize_images"   # folder containing the cell images
output_json = "Test_cleaned_validated_dataset_singleQA.json"

with open(input_json, "r") as f:
    data = json.load(f)

# Handle both list and dict formats
if isinstance(data, dict):
    data = list(data.values())

images = []
annotations = []

image_id = 1
annotation_id = 1

for sample in data:

    file_name = sample["cell_image"]
    image_path = Path(image_root) / file_name

    # Read image size
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"Could not read {image_path}: {e}")
        continue

    # Find all q1,q2,q3,...
    q_keys = sorted(
        [k for k in sample.keys() if re.fullmatch(r"q\d+", k)],
        key=lambda x: int(x[1:])
    )

    for q_key in q_keys:

        q_num = q_key[1:]
        a_key = f"a{q_num}"

        if a_key not in sample:
            continue

        # Repeat image for each QA pair
        images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"{image_root}/{file_name}"
        })

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "Question": sample[q_key],
            "Answer": sample[a_key],
            "category_id": 1,
            "task": "QA"
        })

        image_id += 1
        annotation_id += 1

coco_data = {
    "images": images,
    "annotations": annotations,
    "categories": [
        {
            "id": 1,
            "name": "QA"
        }
    ]
}

with open(output_json, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"Created {len(images)} image entries")
print(f"Created {len(annotations)} annotation entries")