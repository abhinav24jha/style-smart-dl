import os
import json
from segmenting_clothes_script import image_segmentation
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

list_of_image_paths = []

with open('generated_dataset.json', 'r') as f:
    data = json.load(f)

for datapoint in data:
    list_of_image_paths.append(datapoint["anchor"]["image"]) 
    list_of_image_paths.append(datapoint["positive"]["image"]) 
    list_of_image_paths.append(datapoint["contextual_negative"]["image"]) 
    list_of_image_paths.append(datapoint["fashion_negative"]["image"]) 

# print(list_of_images)
print(len(list_of_image_paths))

processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b3-fashion")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer-b3-fashion")

model.to(device)

vilt_segmented_dataset_images = "vilt_segmented_dataset_images"
base_path = "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/"

for image_path in list_of_image_paths:
    # Remove the leading slash to make it a relative path
    relative_path = image_path.lstrip('/')
    image_path = os.path.join(base_path, relative_path)
    print(image_path)
    image_segmentation(model=model,
                       processor=processor,
                       image_path=image_path,
                       masked_imgs_dir_path=vilt_segmented_dataset_images)
