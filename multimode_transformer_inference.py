from PIL import Image
import itertools
from transformers import ViltProcessor, ViltForImageAndTextRetrieval
from typing import List
import torch
import matplotlib.pyplot as plt

# Load the fine-tuned processor and model
processor = ViltProcessor.from_pretrained("fine_tuned_vilt_style_smart_real_4epochs_new")
model = ViltForImageAndTextRetrieval.from_pretrained("fine_tuned_vilt_style_smart_real_4epochs_new")

# Now the processor and model are ready for inference

def recommend_outfit(image_collection_upper: List[str], 
                     image_collection_lower: List[str], 
                     weather: str, 
                     schedule: str):
    
    best_score = float('-inf')
    best_outfit = None
    outfit_scores = []

    # Generate all combinations of upper and lower body clothing
    for upper_image_path, lower_image_path in itertools.product(image_collection_upper, image_collection_lower):
        try:
            # Process the images and metadata
            upper_image = Image.open(upper_image_path)
            lower_image = Image.open(lower_image_path)
            
            # Ensure both images are in RGB format
            if upper_image.mode != 'RGB':
                upper_image = upper_image.convert('RGB')
            if lower_image.mode != 'RGB':
                lower_image = lower_image.convert('RGB')
            
            # Create the text input (concatenate weather and schedule)
            text = f"Weather: {weather}, Schedule: {schedule}"

             # Get embeddings for upper and lower images separately
            upper_inputs = processor(images=upper_image, text=[text], return_tensors="pt")
            lower_inputs = processor(images=lower_image, text=[text], return_tensors="pt")

            
            # Pass the inputs through the model to get separate outputs
            upper_outputs = model(**upper_inputs)
            lower_outputs = model(**lower_inputs)

             # Combine the embeddings by concatenation
            combined_embedding = torch.cat((upper_outputs.logits, lower_outputs.logits), dim=-1)

            # Use the combined embedding to determine the most appropriate outfit (higher is better)
            score = combined_embedding.mean().item()  # Convert the single value tensor to a Python float
            
            # Store the outfit and its score
            outfit_scores.append((upper_image_path, lower_image_path, score))
            
            # Update the best outfit if this score is higher
            if score > best_score:
                best_score = score
                best_outfit = (upper_image_path, lower_image_path)
        
        except Exception as e:
            print(f"Error processing outfit: {e}")
    
    return outfit_scores, best_outfit, best_score

# Example usage
image_collection_upper = [
    "inference_clothes_segmented/class_UpperBodyClothes_4c7e7c1de69643de121a00534961bcbe.jpg.png", "inference_clothes_segmented/class_UpperBodyClothes_1562d4e7d509f0be982cf848b096f361.jpg.png", "inference_clothes_segmented/class_UpperBodyClothes_423070b60d01b85cc92c36f01b3ea56c.jpg.png", "inference_clothes_segmented/class_UpperBodyClothes_b70707f6ddf616613564c8edfdc0cb91.jpg.png", "inference_clothes_segmented/class_UpperBodyClothes_d038dad09ccc05d6f76c89dd11126db5.jpg.png"
]

image_collection_lower = [
    "inference_clothes_segmented/class_LowerBodyClothes_4c7e7c1de69643de121a00534961bcbe.jpg.png", "inference_clothes_segmented/class_LowerBodyClothes_1562d4e7d509f0be982cf848b096f361.jpg.png", "inference_clothes_segmented/class_LowerBodyClothes_423070b60d01b85cc92c36f01b3ea56c.jpg.png", "inference_clothes_segmented/class_LowerBodyClothes_b70707f6ddf616613564c8edfdc0cb91.jpg.png", 
    "inference_clothes_segmented/class_LowerBodyClothes_d038dad09ccc05d6f76c89dd11126db5.jpg.png"
]



weather = "Mild"
schedule = "Formal"

# Get the recommended outfit and all scores
outfit_scores, recommended_outfit, recommendation_score = recommend_outfit(image_collection_upper, image_collection_lower, weather, schedule)

# Print all outfits with their scores
print("All outfit combinations with scores:")
for outfit in outfit_scores:
    print(f"Upper: {outfit[0]}, Lower: {outfit[1]}, Score: {outfit[2]}")

# Print the recommended outfit
print(f"\nRecommended outfit: Upper: {recommended_outfit[0]}, Lower: {recommended_outfit[1]}, Score: {recommendation_score}")

# Load the images using PIL
recommended_upper_image = Image.open(recommended_outfit[0])
recommended_lower_image = Image.open(recommended_outfit[1])

# Set up the Matplotlib figure and axes
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# Display the upper image
axes[0].imshow(recommended_upper_image)
axes[0].set_title("Upper Body Clothing")
axes[0].axis("off")  # Hide axis

# Display the lower image
axes[1].imshow(recommended_lower_image)
axes[1].set_title("Lower Body Clothing")
axes[1].axis("off")  # Hide axis

# Print the recommended outfit details
print(f"\nRecommended outfit: Upper: {recommended_outfit[0]}, Lower: {recommended_outfit[1]}, Score: {recommendation_score}")

plt.show()