from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import os


processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b3-fashion")
model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer-b3-fashion")
model.eval()


def image_segmentation(model: AutoModelForSemanticSegmentation,
                       processor: SegformerImageProcessor,
                       image_path: str,
                       masked_imgs_dir_path: str,
                       confidence_threshold: float = 0.7,
                       background_color=(255, 255, 255, 0)):

    image = Image.open(image_path)

    os.makedirs(masked_imgs_dir_path, exist_ok=True)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu()

        # Apply softmax to get probabilities
        # probs = nn.functional.softmax(logits, dim=1)

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]
        # max_probs = upsampled_probs.max(dim=1)[0]

    pred_seg_numpy = pred_seg.numpy()
    # max_probs_numpy = max_probs.numpy()

    # Define the new combined class groupings
    class_groups = {
        "UpperBodyClothes": [
            1,  # Shirt, Blouse
            2,  # Top, T-shirt, Sweatshirt
            3,  # Sweater
            4,  # Cardigan
            5,  # Jacket
            6,  # Vest
            10, # Coat
            11, # Dress
            12, # Jumpsuit
            26, # Scarf
            28, # Hood
            29, # Collar
            30, # Lapel
            31, # Epaulette
            32, # Sleeve
            33, # Pocket (shared)
            34, # Neckline
            41, # Fringe
            42, # Ribbon
            44, # Ruffle
            45, # Sequin
            46, # Tassel
            13, # Cape
            17, # Tie
            43, # Rivet
            37, # Applique
            38, # Bead
        ],
        "LowerBodyClothes": [
            7,  # Pants
            8,  # Shorts
            9,  # Skirt
            20, # Belt
            21, # Leg Warmer
            22, # Tights, Stockings
            33, # Pocket (shared)
            35, # Buckle
            36, # Zipper (shared between Upper and Lower)
            23, # Sock
            43, # Rivet
        ]
    }

    for group_name, class_labels in class_groups.items():
        # if class_label in class_labels:
        # binary mask for region of interest
        binary_mask = np.isin(pred_seg_numpy, class_labels).astype(np.uint8)

        # Apply confidence threshold
        # confident_pixels = max_probs_numpy > confidence_threshold
        # binary_mask = binary_mask & confident_pixels

        # Check if there are any pixels in the mask after applying the threshold
        # if not np.any(binary_mask):
        #     print(f"No confident pixels found for {group_name}, skipping...")
        #     continue

        # to apply the binary mask to the orginal image, we convert the PIL image into a np array
        image_np = np.array(image)

         # ensuring the binary_mask has the correct shape to match with the image
         # (1, height, width) --> (height, width, 1)
        # binary_mask = binary_mask.reshape(binary_mask.shape[1], binary_mask.shape[2], 1)

        if len(binary_mask.shape) == 2 and len(image_np.shape) == 3:
            # shape of binary_mask is (height1, width1) and that of image_np is (height, width, 3)
            binary_mask = binary_mask[:, :, np.newaxis] # binary_mask -> (height1, width1, 1)

            #we need this binary_mask for each of the three color channels
            binary_mask = np.repeat(binary_mask, 3, axis=2) # binary_mask -> (height1, width1, 3)

        # now we can apply the binary mask to the original image
        masked_image_np = image_np * binary_mask


        # Now convert to RGBA to add transparency to non-clothing areas
        masked_image_np_rgba = np.zeros((masked_image_np.shape[0], masked_image_np.shape[1], 4), dtype=np.uint8)
        masked_image_np_rgba[:, :, :3] = masked_image_np  # Set RGB channels
        masked_image_np_rgba[:, :, 3] = np.where(binary_mask[:, :, 0] == 1, 255, 0)  # Set alpha channel


        # convering this masked image back to PIL format to display
        # masked_image = Image.fromarray(masked_image_np.astype(np.uint8))
        masked_image = Image.fromarray(masked_image_np_rgba, 'RGBA')

        #adding the masked_img to the masked_imgs_dir
        image_name = os.path.basename(image_path)
        
        masked_image_path = os.path.join(masked_imgs_dir_path, f'class_{group_name}_{image_name}.png')
        masked_image.save(masked_image_path)

        print(f'Saved masked image for class {group_name}_{image_name}')



# image_path = 'fashion_disaster_images/Screenshot 2024-08-27 at 9.38.53 AM.png'
# masked_imgs_dir_path = 'proper_segmented_clothes'

# image_segmentation(model=model,
#                    processor=processor,
#                    image_path=image_path,
#                    masked_imgs_dir_path=masked_imgs_dir_path)