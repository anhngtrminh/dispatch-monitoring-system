import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam
from tqdm import tqdm

from inference import DetectionClassificationPipeline

# ----------------------------
# Config
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4

CLASSIFIERS = []
BASE_PATH = "data/classification"

# Determine which classifiers have valid feedback (excluding 'unknown' or empty folders)
for obj_type in ["tray", "dish"]:
    correction_dir = os.path.join(BASE_PATH, obj_type)
    if not os.path.exists(correction_dir):
        continue
    subdirs = [os.path.join(correction_dir, d) for d in os.listdir(correction_dir)]
    valid_data = any(os.path.isdir(d) and len(os.listdir(d)) > 0 for d in subdirs)
    if valid_data:
        CLASSIFIERS.append(obj_type)

# ----------------------------
# Training Function
# ----------------------------
def train_classifier_from_existing(classifier_type):
    print(f"\n🔁 Fine-tuning model for: {classifier_type}")

    correction_dir = os.path.join(BASE_PATH, classifier_type)
    model_load_path = f"models/{classifier_type}_classifier.pt"
    model_save_path = f"models/{classifier_type}_classifier_new.pt"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(correction_dir, transform=transform)
    if len(dataset) == 0:
        print(f"⚠️ No data found in {correction_dir}. Skipping.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    class_names = dataset.classes
    num_classes = len(class_names)

    # Load existing model if available
    model = models.efficientnet_b1(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

    # from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    # model = efficientnet_b1(weights=None)

    if os.path.exists(model_load_path):
        print(f"✅ Loading existing model from {model_load_path}")
        checkpoint = torch.load(model_load_path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        model.eval().to(DEVICE)
        # model.load_state_dict(torch.load(model_load_path, map_location=DEVICE))
    else:
        print("⚠️ No base model found. Training from scratch.")

    model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Fine-tuned model saved to {model_save_path}")


# ----------------------------
# Run applicable classifiers
# ----------------------------
if __name__ == "__main__":
    if not CLASSIFIERS:
        print("ℹ️ No valid feedback for tray or dish. Nothing to fine-tune.")
    for classifier in CLASSIFIERS:
        train_classifier_from_existing(classifier)
