import torch
import torchvision
from torchvision import transforms
from typing import Dict, List
import os

from binary_classification_model_files.inference_one_at_time import pred_and_plot_image

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names

def batch_pred(model_path: str,
               input_folder_path: str,
               output_path: str,
               transform: torchvision.transforms = None,
               device: torch.device = device):
    
    class_names = {0: "No Clothes", 1: "Yes Clothes"}
    
    # Loading in the trained model
    model = torchvision.models.efficientnet_b0(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, out_features=1, bias=True),
        torch.nn.Sigmoid()
    )

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print("Error: Model file not found. Please check the file path.")
        exit(1)

    model.to(device)

    input_folder_path = input_folder_path
    os.makedirs(input_folder_path, exist_ok=True)
    images_files_list = os.listdir(input_folder_path)
    images_files_list_paths = []

    for image_file in images_files_list:
        images_files_list_paths.append(os.path.join(input_folder_path, image_file))

    for image_path in images_files_list_paths:
        pred_and_plot_image(model=model, 
                            image_path=image_path,
                            class_names=class_names,
                            output_path=output_path,
                            transform=transform,
                            device=device)

def main():
    input_folder_path = "input_folder"
    output_path = "yes_clothes_folder"
    model_path = "models/og_binary_classification_model_state_dict.pt"

    # Define your inference dataset and data loader
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_pred(model_path=model_path,
               input_folder_path=input_folder_path,
               output_path=output_path,
               transform=transform,
               device=device)

if __name__ == "__main__":
    main()