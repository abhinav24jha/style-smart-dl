from flask import Flask, request, jsonify
from PIL import Image
import itertools
from transformers import ViltProcessor, ViltForImageAndTextRetrieval
from typing import List
import torch
import base64
import io

app = Flask(__name__)

# Load the fine-tuned processor and model
processor = ViltProcessor.from_pretrained("fine_tuned_vilt_style_smart_real_4epochs_new")
model = ViltForImageAndTextRetrieval.from_pretrained("fine_tuned_vilt_style_smart_real_4epochs_new")

# In-memory storage for outfit scores
outfit_scores_global = []

def recommend_outfit(image_collection_upper: List[str], 
                     image_collection_lower: List[str], 
                     weather: str, 
                     schedule: str):
    global outfit_scores_global
    best_score = float('-inf')
    best_outfit = None
    outfit_scores = []

    # Generate all combinations of upper and lower body clothing
    for upper_image_base64, lower_image_base64 in itertools.product(image_collection_upper, image_collection_lower):
        try:
            # Decode base64 images
            upper_image = Image.open(io.BytesIO(base64.b64decode(upper_image_base64.split(",")[1])))
            lower_image = Image.open(io.BytesIO(base64.b64decode(lower_image_base64.split(",")[1])))

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
            outfit_scores.append((upper_image_base64, lower_image_base64, score))

            # Update the best outfit if this score is higher
            if score > best_score:
                best_score = score
                best_outfit = (upper_image_base64, lower_image_base64)

        except Exception as e:
            print(f"Error processing outfit: {e}")

    # Sort outfits by score in descending order
    outfit_scores.sort(key=lambda x: x[2], reverse=True)
    outfit_scores_global = outfit_scores  # Store globally for future access

    return outfit_scores, best_outfit, best_score

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    tops = data.get('tops')
    bottoms = data.get('bottoms')
    weather = data.get('weather')
    schedule = data.get('schedule')

    if not tops or not bottoms:
        return jsonify({"error": "Tops and bottoms are required"}), 400

    if not weather or not schedule:
        return jsonify({"error": "Weather and schedule labels are required"}), 400

    # Run the recommendation logic
    outfit_scores, recommended_outfit, recommendation_score = recommend_outfit(tops, bottoms, weather, schedule)

    # Prepare the response with recommended images
    recommended_upper_image = recommended_outfit[0]
    recommended_lower_image = recommended_outfit[1]

    return jsonify({
        'recommended_upper_image': recommended_upper_image,
        'recommended_lower_image': recommended_lower_image,
        'weather': weather,
        'schedule': schedule,
        'outfit_scores': outfit_scores,  # Send all outfit scores for client-side handling
    })

@app.route('/next_recommendation', methods=['POST'])
def next_recommendation():
    data = request.json
    current_index = data.get('current_index', 0)

    # Check if the index is valid
    if current_index + 1 < len(outfit_scores_global):
        next_outfit = outfit_scores_global[current_index + 1]
        return jsonify({
            'recommended_upper_image': next_outfit[0],
            'recommended_lower_image': next_outfit[1],
            'next_index': current_index + 1
        })
    else:
        return jsonify({"error": "No more recommendations available"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)