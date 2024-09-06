import os 
import shutil
import random

def split_images_train_test_split(source_dir, 
                                  dest_train_dir,
                                  dest_test_dir, 
                                  num_images_to_select = 10000, 
                                  train_ratio = 0.8):

    os.makedirs(dest_train_dir, exist_ok=True)
    os.makedirs(dest_test_dir, exist_ok=True)

    # List of all the image files in the source_dir
    images = [f for f in os.listdir(source_dir) if f.endswith('jpg')]
    total_images_source_dir = len(images)

    if total_images_source_dir < num_images_to_select:
        print(f"Not enough images, the number of images available is {len(images)}")
        return

    #randomly selecting 'num_images_to_select' number of images from the list
    random_images_selection = random.sample(images, num_images_to_select)

    #further shuffling
    random.shuffle(random_images_selection)

    #split idx
    split_idx = int(num_images_to_select * train_ratio)

    #splitting the images into the respective testing & training sets
    train_images = random_images_selection[:split_idx]
    test_images = random_images_selection[split_idx:]

    #copying the train_imgs to the dest_train_dir
    for image in train_images:
        src_path = os.path.join(source_dir, image)  
        dest_path = os.path.join(dest_train_dir, image)
        shutil.copy(src_path, dest_path)
    
    #similarly for test_imgs to the dest_test_dir
    for image in test_images:
        src_path = os.path.join(source_dir, image)  
        dest_path = os.path.join(dest_test_dir, image)  
        shutil.copy(src_path, dest_path)

    print(f"Training Split Processed! {len(train_images)} images copied to {dest_train_dir}")
    print(f"Testing Split Processed! {len(test_images)} images copied to {dest_test_dir}")