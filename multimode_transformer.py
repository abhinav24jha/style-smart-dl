import torch
from transformers import ViltProcessor, ViltForImageAndTextRetrieval
from PIL import Image, ImageOps
import json

# Load the ViLT model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

def triplet_loss(anchor, positive, contextual_negative, fashion_negative, margin=0.5, smoothing_factor=0.1):
    # Calculate distances
    pos_distance = torch.pairwise_distance(anchor, positive)
    contextual_neg_distance = torch.pairwise_distance(anchor, contextual_negative)
    fashion_neg_distance = torch.pairwise_distance(anchor, fashion_negative)

    # Target positive distance is not 0 but smoothing_factor
    smooth_pos_distance = torch.relu(pos_distance - smoothing_factor)

    # Apply the triplet loss
    # target negative distance is -> margin - smoothing_factor
    contextual_loss = torch.relu(smooth_pos_distance - contextual_neg_distance + margin - smoothing_factor)
    fashion_loss = torch.relu(smooth_pos_distance - fashion_neg_distance + margin - smoothing_factor)

    # Combine the losses
    loss = torch.mean(contextual_loss + fashion_loss)
    return loss, pos_distance, contextual_neg_distance, fashion_neg_distance


# Function to process each image (upper or lower) and get their individual embeddings
def get_single_image_embedding(image_path, text):
    try:
        # Open image
        image = Image.open(image_path)
        
        # Ensure the image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process the image and text using the ViLT processor
        inputs = processor(images=image, text=[text], return_tensors="pt")

        # Get the logits from the model's output
        outputs = model(**inputs)

        # Use logits as the embedding
        embedding = outputs.logits

        return embedding

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Function to combine embeddings from upper and lower images into a single outfit embedding
def get_combined_embedding(upper_image_path, lower_image_path, weather, schedule):
    # Concatenate the weather and schedule metadata into text
    text = f"Weather: {weather}, Schedule: {schedule}"

    # Get embeddings for upper and lower images separately
    upper_embedding = get_single_image_embedding(upper_image_path, text)
    lower_embedding = get_single_image_embedding(lower_image_path, text)

    # Ensure embeddings are valid
    if upper_embedding is None or lower_embedding is None:
        return None

    # Combine the embeddings by concatenation
    combined_embedding = torch.cat((upper_embedding, lower_embedding), dim=-1)  # Concatenate embeddings

    return combined_embedding

# Fine-tuning loop
def fine_tune(dataset, num_epochs=10, learning_rate=1e-6):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        pos_distances = []
        neg_distances = []
        for item in dataset:
            # Get embeddings for the anchor, positive, and both types of negatives
            anchor_embed = get_combined_embedding(item["anchor"]["image_upper_cloth"], 
                                                  item["anchor"]["image_lower_cloth"], 
                                                  item["anchor"]["weather"], 
                                                  item["anchor"]["schedule"])
            
            positive_embed = get_combined_embedding(item["positive"]["image_upper_cloth"], 
                                                    item["positive"]["image_lower_cloth"], 
                                                    item["positive"]["weather"], 
                                                    item["positive"]["schedule"])
            
            contextual_neg_embed = get_combined_embedding(item["contextual_negative"]["image_upper_cloth"], 
                                                          item["contextual_negative"]["image_lower_cloth"], 
                                                          item["contextual_negative"]["weather"], 
                                                          item["contextual_negative"]["schedule"])
            
            fashion_neg_embed = get_combined_embedding(item["fashion_negative"]["image_upper_cloth"], 
                                                       item["fashion_negative"]["image_lower_cloth"], 
                                                       item["fashion_negative"]["weather"], item["fashion_negative"]["schedule"])
            
            # Check for None embeddings and skip if necessary
            if None in [anchor_embed, positive_embed, contextual_neg_embed, fashion_neg_embed]:
                print(f"Skipping a batch due to missing embeddings.")
                continue  # Skip this item if any embedding failed

            # Calculate contrastive loss
            loss, pos_distance, contextual_neg_distance, fashion_neg_distance = triplet_loss(anchor_embed, positive_embed, contextual_neg_embed, fashion_neg_embed)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pos_distances.append(pos_distance.item())  # No need for .mean() since it's a single value
            neg_distances.append(min(contextual_neg_distance.item(), fashion_neg_distance.item()))

            # Calculate accuracy for this batch
            if pos_distance.item() < min(contextual_neg_distance.item(), fashion_neg_distance.item()):
                correct_predictions += 1
            total_samples += 1

        # Calculate epoch metrics
        accuracy = correct_predictions / total_samples
        mean_pos_distance = sum(pos_distances) / len(pos_distances)
        mean_neg_distance = sum(neg_distances) / len(neg_distances)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataset)}, Accuracy: {accuracy:.4f}", 
              f"Mean Positive Distance: {mean_pos_distance:.4f}, Mean Negative Distance: {mean_neg_distance:.4f}")

# Sample dataset
# dataset = [
#     {
#         "anchor": {
#             "image_upper_cloth": "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/proper_segmented_clothes/batch_1/class_UpperBodyClothes_4beac1a2ceb6ff197cd14c57547d5356.jpg.png",
#             "image_lower_cloth": "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/proper_segmented_clothes/batch_1/class_LowerBodyClothes_4beac1a2ceb6ff197cd14c57547d5356.jpg.png",
#             "weather": "Warm",
#             "schedule": "Casual"
#         },
#         "positive": {
#             "image_upper_cloth": "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/proper_segmented_clothes/batch_1/class_UpperBodyClothes_images.jpg.png",
#             "image_lower_cloth": "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/proper_segmented_clothes/batch_1/class_LowerBodyClothes_images.jpg.png",
#             "weather": "Warm",
#             "schedule": "Casual"
#         },
#         "contextual_negative": {
#             "image_upper_cloth": "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/proper_segmented_clothes/batch_1/class_UpperBodyClothes_Screenshot 2024-08-25 at 8.33.33\u202fAM.png.png",
#             "image_lower_cloth": "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/proper_segmented_clothes/batch_1/class_LowerBodyClothes_Screenshot 2024-08-25 at 8.33.33\u202fAM.png.png",
#             "weather": "Warm", 
#             "schedule": "Sporty"
#         },
#         "fashion_negative": {
#             "image_upper_cloth": "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/proper_segmented_clothes/class_UpperBodyClothes_Screenshot 2024-08-27 at 9.38.53\u202fAM.png.png",
#             "image_lower_cloth": "/Users/abhinavjha/Desktop/binaryClassification_&_segmentation/proper_segmented_clothes/class_LowerBodyClothes_Screenshot 2024-08-27 at 9.38.53\u202fAM.png.png",
#             "weather": "Warm",  
#             "schedule": "Casual" 
#         }
#     }
# ]

dataset_file_path = "modified_generated_dataset.json"  # Replace with your actual file path
with open(dataset_file_path, 'r') as f:
    data = json.load(f)

dataset = data['dataset']

# Fine-tune the model with the dataset
fine_tune(dataset, num_epochs=4)  # Run for 1 epoch for testing

# Save the fine-tuned model
model.save_pretrained("fine_tuned_vilt_style_smart_real_4epochs_new")
processor.save_pretrained("fine_tuned_vilt_style_smart_real_4epochs_new")

print("Fine-tuning complete. Model saved.")

"""
Epoch 1/3, Loss: 0.9557724993417759, Accuracy: 0.6238 Mean Positive Distance: 1.7552, Mean Negative Distance: 1.9884
Epoch 2/3, Loss: 0.0905895188893422, Accuracy: 0.9703 Mean Positive Distance: 0.9042, Mean Negative Distance: 2.3978
Epoch 3/3, Loss: 0.00012124824051809783, Accuracy: 1.0000 Mean Positive Distance: 0.8878, Mean Negative Distance: 2.7586
Fine-tuning complete. Model saved.
"""