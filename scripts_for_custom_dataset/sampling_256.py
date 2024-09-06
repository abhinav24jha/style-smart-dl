# The goal of this file, is to write a script which extracts 1300 random annotated .json files, stores the 
#  corresponding image names to these annotated .json files, and does two things:
#   1. Purpose of storing the image names in a list, is to write another script, to put those images in a separate folder
#   2. The random 1300 random annoted files are to be adjusted to contain useful information for my task
#           - imagefile_name
#           - category_id
#           - category_name
#           - bounding_box [potentially will allow the neural net to identify clothes in images better] 

import os
import json
import random
from tqdm import tqdm

#Directory Paths

annotations_dir = '/Users/abhinavjha/Desktop/train/annos'
new_annotations_dir = '/Users/abhinavjha/Desktop/collection/annotation'
new_imagenames_path = '/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/scripts_for_custom_dataset/new_imagenames.txt'

os.makedirs(new_annotations_dir, exist_ok=True)

#The logic behind this number was initally meant for 100 per category, but we're doing random_sampling instead of
#balanced sampling, because of potential overtraining as most of the images have multiple clothing items, putting
#each image in multiple categories

sample_size = 256 

#getting all the .json files in the annotations_dir in a list

all_annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
# print(len(all_annotation_files))

# randomly sampling 1300 of them
sampled_files = random.sample(all_annotation_files, sample_size)
# print(len(sampled_files))

image_filenames = []

total_files_new_annotations_dir = len([f for f in os.listdir(new_annotations_dir) if f.endswith('.json')])

def main():

    for annotation_file in tqdm(sampled_files):
        annotation_file_path = os.path.join(annotations_dir, annotation_file)

        with open(annotation_file_path, 'r') as f:
            og_annotated_file = json.load(f)

        image_name = annotation_file.replace('json', 'jpg')
        image_filenames.append(image_name)

        cleaned_annotated_file = {
            'image_name': image_name,
            'items': []
        }

        for key in og_annotated_file.keys():
            if key.startswith('item'):
                key_data = og_annotated_file.get(key, "The key starting with item doesn't exist")
                # print(type(key_data))
                # print(key_data)
                category_id = key_data.get('category_id', "The key category_id is not found.")
                category_name = key_data.get('category_name', "The key category_name is not found")
                bounding_box = key_data.get('bounding_box', "The key bounding_box is not found")

            # we want to add a particular clothing from the image only if we are able to get all three pieces of
            # information about the image, the category_id & category_name & bounding_box

                if (category_id != "The key category_id is not found.") and (category_name != "The key category_name is not found") and (bounding_box != "The key bounding_box is not found"):

                    cleaned_annotated_file['items'].append({
                        "category_id": category_id,
                        "category_name": category_name,
                        "binary_class": 1, # 1 indicates clothes in img, 0 indicates no clothes in img
                        "bounding_box": bounding_box
                    })


            new_annotation_file_path = os.path.join(new_annotations_dir, annotation_file)

            with open(new_annotation_file_path, 'w') as outfile:
                json.dump(cleaned_annotated_file, outfile, indent=4)


    # When this point is reach all the randomly sampled annotated .json files should have been 
    # formatted accordingly, and we should have a list with the corresponding image_names

    # Now we're going to write the list of image_names into a .txt file
    with open("/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/scripts_for_custom_dataset/new_imagenames.txt", 'w') as f:
        for image_filename in image_filenames:
            f.write(f"{image_filename}\n")

if __name__ == "__main__":
    # Ensures re-running this program doesn't just keep adding batches of 1300, rather it checks if the
    # directory has no files then it decides to run main, otherwise doesn't
    if(total_files_new_annotations_dir == 0):
        main()
    else:
        pass