"""
Introduce this type of folder structure, when you want to deal with other tasks, not just a simple binary
classification on the bases of a simple label, suppose you wanted to include the bounding boxes of the various
things in the images then use this structure:-

Desktop
    no_clothes [This folder like the name suggests contains the train/test split, of images with binary_class: 0]
        test
            no_clothes_annotations_test
            no_clothes_images_test
        train
            no_clothes_annotations_train
            no_clothes_images_train

    new_train [This folder contains the train/test split, of images with binary_class: 1]
        test
            new_annotations_test
            new_images_test
        train
            new_annotations_train
            new_annotations_train
"""

"""
The folder structure we're going to use is going to take benefit of ImageFolder which automatically assigns
Class Label, not on semantic meaning, but rather which ever comes before in alphabetical case gets assigned the
label:0, while the other as label:1

Desktop
    binary_classification_model_data
        train
            yes_clothes [binary_class: 1]
            no_clothes [alphabetically before, so binary_class: 0]
        test
            yes_clothes [binary_class: 1]
            no_clothes [alphabetically before, so binary_class: 0]
"""


import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from training_loop import train
from saving_the_model import save_model
from plotting_loss_curves import plot_loss_curves
from train_test_split_script import split_images_train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

#for the label -> "no_clothes"
split_images_train_test_split(source_dir='/Users/abhinavjha/Desktop/PASS_dataset/7',
                              dest_train_dir='/Users/abhinavjha/Desktop/binary_classification_model_data_initial/train/no_clothes',
                              dest_test_dir='/Users/abhinavjha/Desktop/binary_classification_model_data_initial/test/no_clothes',
                              num_images_to_select=1250,
                              train_ratio=0.8
)

#for the label -> "yes_clothes"
split_images_train_test_split(source_dir='/Users/abhinavjha/Desktop/train/image',
                              dest_train_dir='/Users/abhinavjha/Desktop/binary_classification_model_data_initial/train/yes_clothes',
                              dest_test_dir='/Users/abhinavjha/Desktop/binary_classification_model_data_initial/test/yes_clothes',
                              num_images_to_select=1250,
                              train_ratio=0.8
)

train_dir = '/Users/abhinavjha/Desktop/binary_classification_model_data_initial/train'
test_dir = '/Users/abhinavjha/Desktop/binary_classification_model_data_initial/test'

#using ImageFolder we create the train/test datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)

#create the train/test dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Load the pre-trained EfficientNet Model [EfficientNet-B0]
# model = models.efficientnet_b0(pretrained=True) # Turns out the method is deprecated
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights).to(device=device)

# freeze the base-layers, in order to retain the pre-trained weights
for param in model.features.parameters():
    param.requires_grad = False

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=1,
                    bias=True),
    torch.nn.Sigmoid()
).to(device=device)

# Using binary cross entropy loss
loss_fn = nn.BCELoss()

# this optimizer will only update the weights in the classifier layer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10

results = train(model=model, 
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=epochs,
                device=device)

print(results)

# Save the trained model
save_model(model=model,
           target_dir="models",  # Directory where the model will be saved
           model_name="og_binary_classification_model_state_dict.pt")  # Model filename


plot_loss_curves(results=results)

"""
With some experiments, done on the model in google_colab, what seems to work really well, 
is augmenting data in the training dataset, and then increasing the learning rate from 1e-4 to 1e-3

The results look like
 10%|█         | 1/10 Epoch: 1 | train_loss: 0.6306 | train_acc: 0.6476 | test_loss: 0.5064 | test_acc: 0.9375
 20%|██        | 2/10 Epoch: 2 | train_loss: 0.4838 | train_acc: 0.9062 | test_loss: 0.4215 | test_acc: 0.9732
 30%|███       | 3/10 Epoch: 3 | train_loss: 0.4145 | train_acc: 0.9024 | test_loss: 0.3465 | test_acc: 0.9821
 40%|████      | 4/10 Epoch: 4 | train_loss: 0.3560 | train_acc: 0.9106 | test_loss: 0.3026 | test_acc: 0.9911
 50%|█████     | 5/10 Epoch: 5 | train_loss: 0.3192 | train_acc: 0.9130 | test_loss: 0.2610 | test_acc: 0.9911
 60%|██████    | 6/10 Epoch: 6 | train_loss: 0.2916 | train_acc: 0.9231 | test_loss: 0.2395 | test_acc: 0.9911
 70%|███████   | 7/10 Epoch: 7 | train_loss: 0.2796 | train_acc: 0.9312 | test_loss: 0.2062 | test_acc: 0.9911
 80%|████████  | 8/10 Epoch: 8 | train_loss: 0.2796 | train_acc: 0.9010 | test_loss: 0.1966 | test_acc: 0.9911
 90%|█████████ | 9/10 Epoch: 9 | train_loss: 0.2641 | train_acc: 0.9130 | test_loss: 0.1913 | test_acc: 1.0000
100%|██████████| 10/10 Epoch: 10 | train_loss: 0.2265 | train_acc: 0.9409 | test_loss: 0.1789 | test_acc: 0.9911

I've manually downloaded the saved state_dict of the model, from google_colab and put it the models_folder here

Check "modified_model_loss_curves.png" for the corresponding loss curves associated to the above

Need to definetely, train the model, on a lot more data and see if the same holds up
"""