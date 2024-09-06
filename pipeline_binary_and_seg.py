from binary_classification_batch_inference import batch_pred
from segmentation_batch import batch_segmentation
import torch
from torchvision import transforms
import os


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = (224, 224)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # yes_clothes_folder -> this is going to do binary_classification on all the images, and put the images
    # which have clothes in them in the "yes_clothes_folder"

    model_path = "models/og_binary_classification_model_state_dict.pt"
    input_folder_path = "input_folder"
    yes_clothes_folder_path = "yes_clothes_folder"

    batch_pred(model_path=model_path,
               input_folder_path=input_folder_path,
               output_path=yes_clothes_folder_path,
               transform=transform,
               device=device)
    
    # now we'll take all the images inside of "yes_clothes_folder" and segment them and put it into
    # the folder called "segmented_yes_clothes_folder"

    seg_input_folder = "yes_clothes_folder"
    seg_output_folder = "segmented_yes_clothes_folder"

    batch_segmentation(input_folder_path=seg_input_folder,
                       output_folder_path=seg_output_folder,
                       device=device)
    

if __name__ == "__main__":
    main()