# This is how the script should work

"""
{
  "dataset": [
    {
      "image": "4beac1a2ceb6ff197cd14c57547d5356.jpg",
      "weather": "Warm",
      "schedule": "Casual"
    },
    ..............
    ............
    {
      "image": "Sustainably+Chic+_+Sustainable+Fashion+Blog+_+Best+Sustainable+Mens+Clothing+Brands+.jpg",
      "weather": "Warm",
      "schedule": "Outdoor"
    }
  ]
}


Use the folder where all these images are present, pass them through the segmenting_batch script, 
and now in another folder, you will get all of the above images segmented 

the type would be like this for each img:
    class_LowerBodyClothes_{img_name}.png
    class_UpperBodyClothes_{img_name}.png


now using the list of images extracted from the json earlier, create a new .json file with the structure where image tag is replaced by 
"image_upper_cloth" -> corresponds to -> class_LowerBodyClothes_{img_name}.png
"image_lower_cloth" -> corresponds to -> class_UpperBodyClothes_{img_name}.png

and keep the rest the same
"""

import json
from segmentation_batch import batch_segmentation
import torch 
import os

# Step 1 -> Segmenting the Images in the Input Folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_number = 5

yes_clothes_folder_path = f"men's clothe/batch_{batch_number}"
masked_imgs_dir_path = f"/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/proper_segmented_clothes/batch_{batch_number}"

batch_segmentation(input_folder_path=yes_clothes_folder_path,
                    output_folder_path=masked_imgs_dir_path,
                    device=device)

# Step 2 -> Creating a new json file

with open(f"raw_dataset_multimode_transformers/raw_dataset_{batch_number}.json", 'r') as f:
    data = json.load(f)

new_data = {"dataset": []}

for image_data in data.get('dataset'):
    img_name = image_data.get('image')

    # Construct the new file names for segmented images
    upper_cloth_img = f"class_UpperBodyClothes_{img_name}.png"
    lower_cloth_img = f"class_LowerBodyClothes_{img_name}.png"

    upper_cloth_img_path = os.path.join(masked_imgs_dir_path, upper_cloth_img)
    lower_cloth_img_path = os.path.join(masked_imgs_dir_path, lower_cloth_img)

    # Construct the new JSON entry
    new_entry = {
        "image_upper_cloth": upper_cloth_img_path,
        "image_lower_cloth": lower_cloth_img_path,
        "weather": image_data.get("weather"),
        "schedule": image_data.get("schedule")
    }

    # Add the new entry to the dataset list
    new_data['dataset'].append(new_entry)

# Step 3 -> Write the new data to a new JSON file
with open(f"modified_dataset_multimode_transformers/modified_dataset_{batch_number}.json", 'w') as f:
    json.dump(new_data, f, indent=4)

print("New JSON file has been created with the updated structure.")



"""
Perfect, you're going to be a professional dataset creator for machine learning models, and you're responsible for creating a JSON file, with the corresponding images making up the outfit, corresponding to the weather, and schedule information, ensure that the overall outfit makes fashion sense, and incorporates the weather and schedule information

You should have absolute perfection while creating this dataset, it is to be reviewed by an industry expert, Note that the images that you are being provided are already complete outfits, so dont end up combining two full outfits together as that would not make any sense

Ensure the image_names used are accurate to their actual names, make sure the outfit recommendation is perfect

Make sure you use these labels

Generalized Weather Labels:
    Cold: (e.g., below 10°C)
    Cool: (e.g., 10-18°C)
    Mild: (e.g., 18-25°C)
    Warm: (e.g., 25-30°C)
    Hot: (e.g., above 30°C)
    Rainy: Regardless of temperature, if rain is expected.
    Windy: For days where wind conditions are a major factor.
    Sunny: Clear skies with moderate to hot temperatures.

Generalized Schedule Labels:
    Formal: Business meetings, formal events, weddings, etc.
    Casual: Day-to-day activities, running errands, going out for coffee.
    Sporty: Physical activities like gym sessions, jogging, or hiking.
    Outdoor: Activities that take place outside, such as walks, outdoor dining, etc.
    Evening: Events in the evening that are semi-formal or more social in nature.
    Relaxed: Lounging at home or laid-back activities like movie nights.

"I think your weather associations aren't correct who exactly in their right mind would wanna wear formals when it is b/w 25 to 30 " -> Keep this in mind, this is where you made a slight mistake before

The structure of the JSON FILE should follow 

{
  "dataset": [
    {
      "image": "4beac1a2ceb6ff197cd14c57547d5356.jpg",
      "weather": "Warm",
      "schedule": "Casual"
    },
    {
      "image": "14a69ae13f20f3c62aab8adecfb5d239.jpg",
      "weather": "Cool",
      "schedule": "Outdoor"
    },
    {
      "image": "a8e411333a3df03b870dbe7cd07ac0dd.jpg",
      "weather": "Cool",
      "schedule": "Casual"
    },
    {
      "image": "coats-data.jpg",
      "weather": "Cold",
      "schedule": "Outdoor"
    },
    {
      "image": "eda5f6b4c87a0c04638a87a7e47d9950.jpg",
      "weather": "Mild",
      "schedule": "Formal"
    },
    {
      "image": "fc1a9f1373809b0ac40d4955e403394a.jpg",
      "weather": "Hot",
      "schedule": "Outdoor"
    },
    {
      "image": "il_fullxfull.4222662857_fpjx.webp",
      "weather": "Mild",
      "schedule": "Formal"
    },
    {
      "image": "images.jpg",
      "weather": "Warm",
      "schedule": "Casual"
    },
    {
      "image": "Screenshot 2024-08-25 at 8.33.33 AM.png",
      "weather": "Warm",
      "schedule": "Sporty"
    },
    {
      "image": "Sustainably+Chic+_+Sustainable+Fashion+Blog+_+Best+Sustainable+Mens+Clothing+Brands+.jpg",
      "weather": "Warm",
      "schedule": "Outdoor"
    }
  ]
}

Prompt to feed in
"""