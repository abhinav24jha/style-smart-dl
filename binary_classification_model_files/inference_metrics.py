import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
from typing import Dict, List, Tuple

from inference_one_at_time import pred_and_plot_image

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_inference_and_metrics(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    class_names: Dict[int, str]
) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """
    Perform batch inference and calculate metrics.

    Args:
        model: The trained model
        eval_loader: DataLoader for evaluation data
        class_names: Dictionary mapping class indices to class names

    Returns:
        Tuple containing lists of images, true labels, and predicted labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu())

    all_preds = [int(x) for x in all_preds]
    all_labels = [int(x) for x in all_labels]

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names.values(), yticklabels=class_names.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    return all_images, all_labels, all_preds

def plot_images_with_predictions(
    images: List[torch.Tensor],
    true_labels: List[int],
    pred_labels: List[int],
    class_names: Dict[int, str],
    n_images: int = 16
) -> None:
    """
    Plot images with their true and predicted labels.

    Args:
        images: List of image tensors
        true_labels: List of true labels
        pred_labels: List of predicted labels
        class_names: Dictionary mapping class indices to class names
        n_images: Number of images to plot
    """
    plt.figure(figsize=(12, 12))
    indices = np.random.choice(len(images), n_images, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(4, 4, i+1)
        plt.imshow(np.transpose(images[idx].numpy(), (1, 2, 0)))
        plt.title(f'True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}')
        plt.axis('off')
    plt.show()

# Define class names
class_names = {0: "No Clothes", 1: "Yes Clothes"}

# Loading in the trained model
model = torchvision.models.efficientnet_b0(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, out_features=1, bias=True),
    torch.nn.Sigmoid()
)

try:
    model.load_state_dict(torch.load('models/og_binary_classification_model_state_dict.pt', map_location=device))
except FileNotFoundError:
    print("Error: Model file not found. Please check the file path.")
    exit(1)

model.to(device)

# Define your evaluation dataset and data loader
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    eval_dataset = datasets.ImageFolder(root='/Users/abhinavjha/Desktop/inference_folder', transform=transform)
except FileNotFoundError:
    print("Error: Inference images directory not found. Please check the directory path.")
    exit(1)

eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Perform batch inference and display performance metrics
all_images, all_labels, all_preds = batch_inference_and_metrics(model, eval_loader, class_names)

# Visualize correctly and incorrectly classified images
correct = [i for i in range(len(all_labels)) if all_labels[i] == all_preds[i]]
incorrect = [i for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]

print("Correctly Classified Images")
pred_and_plot_image([all_images[i] for i in correct], [all_labels[i] for i in correct], [all_preds[i] for i in correct], class_names)

print("Incorrectly Classified Images")
pred_and_plot_image([all_images[i] for i in incorrect], [all_labels[i] for i in incorrect], [all_preds[i] for i in incorrect], class_names)