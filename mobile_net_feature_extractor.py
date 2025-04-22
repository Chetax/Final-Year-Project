import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from google.colab import drive
import numpy as np
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define dataset path
dataset_path = "/content/drive/My Drive/breakhis_augmented"
saved_model_path = "/content/drive/My Drive/mobilenet_best.pth"
csv_output_path = "/content/drive/My Drive/mobilenet_features.csv"

# Define transformations (normalization based on ImageNet stats)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# Load MobileNet model
mobilenet = models.mobilenet_v2(pretrained=False)
num_features = mobilenet.classifier[1].in_features
mobilenet.classifier = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)

# Load saved weights
mobilenet.load_state_dict(torch.load(saved_model_path))
mobilenet.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet.to(device)

# Extract features and save to CSV
features_list = []
labels_list = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        features = mobilenet.features(images)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # Reduce to (batch_size, 1280, 1, 1)
        features = features.view(features.size(0), -1).cpu().numpy()  # Now (batch_size, 1280)
        
        features_list.extend(features)
        labels_list.extend(labels.cpu().numpy())

# Convert to DataFrame
df = pd.DataFrame(features_list)
df['label'] = labels_list

# Convert to float16 to reduce size
df = df.astype(np.float16)

# Save to CSV
df.to_csv(csv_output_path, index=False)
print(f"Features saved to {csv_output_path}")
