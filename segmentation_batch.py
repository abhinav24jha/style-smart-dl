from segmenting_clothes_script import image_segmentation
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import os
import torch


def batch_segmentation(
        input_folder_path: str,
        output_folder_path: str,
        device: torch.device):
    
    processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b3-fashion")
    model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer-b3-fashion")

    model.to(device)
    
    images_files = os.listdir(input_folder_path)
    images_files_list = []

    for image_file in images_files:
        images_files_list.append(os.path.join(input_folder_path, image_file))
    
    for image_path in images_files_list:
        image_segmentation(model=model,
                           processor=processor,
                           image_path=image_path,
                           masked_imgs_dir_path=output_folder_path)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yes_clothes_folder_path = "inference_clothes"
    masked_imgs_dir_path = "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/inference_clothes_segmented"

    batch_segmentation(input_folder_path=yes_clothes_folder_path,
                       output_folder_path=masked_imgs_dir_path,
                       device=device)

if __name__ == "__main__":
    main()