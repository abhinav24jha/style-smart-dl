# The intention of this file is to use the names of the images provided in the new_imagename.txt, copying each
# image from that folder into the new_train/new_image directory

# 256/16 = 16batches, 12 batches (192 images) of data for train, 4 batches (64 images) of data for test

import os
import shutil

image_names_file_path = 'new_imagenames.txt'

original_images_dir = '/Users/abhinavjha/Desktop/train/image'
new_images_dir = '/Users/abhinavjha/Desktop/collection/images'

os.makedirs(new_images_dir, exist_ok=True)

total_files_new_images_dir = len([f for f in os.listdir(new_images_dir) if f.endswith('.jpg')])


def main():
    with open('/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/scripts_for_custom_dataset/new_imagenames.txt', 'r') as f:
        image_filenames = [line.strip() for line in f.readlines()]

    for image_filename in image_filenames:
        source_path = os.path.join(original_images_dir, image_filename)
        destination_path = os.path.join(new_images_dir, image_filename)

        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        else:
            print(f"Image {image_filename} not found in {original_images_dir}")

    print(f"Images copied to {new_images_dir}")

if __name__ == "__main__":
    if(total_files_new_images_dir == 0):
        main()
    else:
        pass