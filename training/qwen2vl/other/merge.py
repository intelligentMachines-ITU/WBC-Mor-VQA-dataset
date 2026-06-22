import json

file1 = "Train_cleaned_validated_dataset_singleQA.json"
file2 = "Q_A_train.json"
output_file = "WBC_leukemia_ATT_VQA.json"

with open(file1, "r") as f:
    coco1 = json.load(f)

with open(file2, "r") as f:
    coco2 = json.load(f)

merged = {
    "images": [],
    "annotations": [],
    "categories": coco1.get("categories", [])
}

# Map filename -> new image id
image_map = {}

new_image_id = 1
new_ann_id = 1

def add_dataset(coco):
    global new_image_id, new_ann_id

    old_to_new_image = {}

    # Process images
    for img in coco["images"]:

        fname = img["file_name"]

        if fname not in image_map:

            image_map[fname] = new_image_id

            merged["images"].append({
                **img,
                "id": new_image_id
            })

            old_to_new_image[img["id"]] = new_image_id

            new_image_id += 1

        else:
            old_to_new_image[img["id"]] = image_map[fname]

    # Process annotations
    for ann in coco["annotations"]:

        new_ann = ann.copy()

        new_ann["id"] = new_ann_id
        new_ann["image_id"] = old_to_new_image[ann["image_id"]]

        merged["annotations"].append(new_ann)

        new_ann_id += 1


add_dataset(coco1)
add_dataset(coco2)

with open(output_file, "w") as f:
    json.dump(merged, f, indent=4)

print("Merged images:", len(merged["images"]))
print("Merged annotations:", len(merged["annotations"]))