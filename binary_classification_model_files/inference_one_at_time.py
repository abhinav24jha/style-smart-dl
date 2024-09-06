import torch
import torchvision 
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: Dict[int, str], 
                        output_path: str,
                        transform: torchvision.transforms = None,
                        device: torch.device = device):
    
    # Opening the image
    img = Image.open(image_path)

    # Convert RGBA images to RGB
    if img.mode != "RGB":
        img = img.convert('RGB')

    # Create transformation for the images if they don't already exist
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Making prediction on the image
    model.to(device=device)
    model.eval()

    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        image_pred = model(transformed_image.to(device))

    # Print confidence score before thresholding
    image_pred_score = image_pred.item()
    image_pred_label = (image_pred > 0.5).float().item()

    # print(image_pred_label)

    if image_pred_label == 1.0:
        yes_clothes_folder_path = output_path
        os.makedirs(yes_clothes_folder_path, exist_ok=True)
        image_name = os.path.basename(image_path)
        full_path = os.path.join(yes_clothes_folder_path, image_name)
        img.save(full_path)

    # Plotting the image with its predicted label and confidence
    # plt.figure()
    # plt.imshow(img)
    # plt.title(f"Pred: {class_names[int(image_pred_label)]} (Confidence: {image_pred_score:.4f})")
    # plt.axis(False)
    # plt.show()


# Define class names
class_names = {0: "No Clothes", 1: "Yes Clothes"}

# loading in our saved model

# essentially the first step, is to get an instance of our model, without the weights, cuz we have that saved
model = models.efficientnet_b0(weights=None)

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, out_features=1, bias=True),
    torch.nn.Sigmoid()
)

model.to(device)

# now we load the saved weights
model.load_state_dict(torch.load('models/og_binary_classification_model_state_dict.pt', map_location=device))

# pred_and_plot_image(model=model,
#                     image_path='inference_images/Screenshot 2024-08-24 at 11.46.34 AM.jpg',
#                     class_names=class_names)