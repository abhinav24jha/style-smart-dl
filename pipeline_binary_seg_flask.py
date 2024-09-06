from flask import Flask, request, jsonify
import os
import torch
from torchvision import transforms
from binary_classification_batch_inference import batch_pred
from segmentation_batch import batch_segmentation
import base64
import tempfile

app = Flask(__name__)

# Device and transformation setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_images(images):
    # Create temporary directories
    with tempfile.TemporaryDirectory() as input_dir, \
        tempfile.TemporaryDirectory() as yes_clothes_dir, \
        tempfile.TemporaryDirectory() as seg_output_dir:
        input_paths = []
        for image in images:
            image_path = os.path.join(input_dir, image.filename)
            image.save(image_path)
            input_paths.append(image_path)
        
        # Run binary classification on images
        batch_pred(
            model_path="models/og_binary_classification_model_state_dict.pt",
            input_folder_path=input_dir,
            output_path=yes_clothes_dir,
            transform=transform,
            device=device
        )
        
        # Run segmentation on classified images
        batch_segmentation(
            input_folder_path=yes_clothes_dir,
            output_folder_path=seg_output_dir,
            device=device
        )
        
        # Categorize processed images
        tops = []
        bottoms = []
        for filename in os.listdir(seg_output_dir):
            file_path = os.path.join(seg_output_dir, filename)
            with open(file_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                # Check the filename to categorize
                if "UpperBodyClothes" in filename:
                    tops.append(f"data:image/jpeg;base64,{encoded_string}")
                elif "LowerBodyClothes" in filename:
                    bottoms.append(f"data:image/jpeg;base64,{encoded_string}")
                
        return tops, bottoms

@app.route('/process', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    images = request.files.getlist('images')
    tops, bottoms = process_images(images)
    
    return jsonify({'tops': tops, 'bottoms': bottoms}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)