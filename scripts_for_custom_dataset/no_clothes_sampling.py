# This script manually creates an annotation .json file for each of the images [no clothes images]

import os
import json
from tqdm import tqdm

no_clothes_images_dir = '/Users/abhinavjha/Desktop/collection/no_clothes_images'
no_clothes_images_annotations_dir = '/Users/abhinavjha/Desktop/collection/no_clothes_annotations'

total_no_clothes_images_annotations = len([f for f in os.listdir(no_clothes_images_annotations_dir) if f.endswith('.json')])

list_no_clothes_images = [f for f in os.listdir(no_clothes_images_dir) if f.endswith('.jpg')]


os.makedirs(no_clothes_images_annotations_dir, exist_ok=True)

def main():
    for no_cloth_image in tqdm(list_no_clothes_images):

        format_annotated_file_data = {
            'image_name': no_cloth_image, 
            'binary_class': 0
        }

        annotation_file_name = no_cloth_image.replace('.jpg', '.json')

        new_annotation_file_path = os.path.join(no_clothes_images_annotations_dir, annotation_file_name)

        with open(new_annotation_file_path, 'w') as outfile:
            json.dump(format_annotated_file_data, outfile, indent=4)

if __name__ == "__main__":
    if(total_no_clothes_images_annotations == 0):
        main()
    else:
        pass