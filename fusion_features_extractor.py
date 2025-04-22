import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import h5py

# Define dataset path
dataset_path = "/content/drive/MyDrive/fyproject/breakhis_aug/Breakhis_400x"

# Define transformations (normalization based on ImageNet stats)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Function to fine-tune MobileNetV2 with last 20 layers unfrozen
def train_mobilenet():
    print("Training with last 20 layers unfrozen using AdamW optimizer...")
    mobilenet = models.mobilenet_v2(pretrained=True)

    for param in mobilenet.features.parameters():
        param.requires_grad = False  # Freeze all layers by default

    # Unfreeze the last 20 layers
    for param in list(mobilenet.features.parameters())[-20:]:
        param.requires_grad = True

    # Replace classifier head
    num_features = mobilenet.classifier[1].in_features
    mobilenet.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(mobilenet.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet.to(device)

    # Training loop
    epochs = 10
    best_accuracy = 0
    for epoch in range(epochs):
        mobilenet.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = mobilenet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        scheduler.step(accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

        # Save the best model in H5 format
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = "/content/drive/MyDrive/mobilenet_best.h5"
            torch.save(mobilenet.state_dict(), model_path)
            print(f"Best model saved to {model_path}")

    return best_accuracy

# Train the model
best_acc = train_mobilenet()
print(f"Final best accuracy: {best_acc:.2f}%")
