import json
import os

with open("generated_dataset.json", 'r') as f:
    data = json.load(f)

new_data = {"dataset": []}

vilt_segmented_dataset_images_path = "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/vilt_segmented_dataset_images"

for entry in data:
    new_entry = {}

    # Iterate through each type in the entry (anchor, positive, contextual_negative, fashion_negative)
    for key in entry.keys():
        image_data = entry[key]
        image_name = image_data.get('image').split('/')[-1]

        # Construct the new file names for segmented images
        upper_cloth_img = f"class_UpperBodyClothes_{image_name}.png"
        lower_cloth_img = f"class_LowerBodyClothes_{image_name}.png"

        upper_cloth_img_path = os.path.join(vilt_segmented_dataset_images_path, upper_cloth_img)
        lower_cloth_img_path = os.path.join(vilt_segmented_dataset_images_path, lower_cloth_img)

        # Add the new paths to the entry
        new_entry[key] = {
            "image_upper_cloth": upper_cloth_img_path,
            "image_lower_cloth": lower_cloth_img_path,
            "weather": image_data.get("weather"),
            "schedule": image_data.get("schedule")
        }
    
    # Add the new entry to the dataset list
    new_data['dataset'].append(new_entry)

# Write the new data to a new JSON file
with open(f"modified_generated_dataset.json", 'w') as f:
    json.dump(new_data, f, indent=4)

print("New JSON file has been created with the updated structure.")
