import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, classification_report
import argparse

def create_data_loaders(data_dir, batch_size=32, val_split=0.2, num_workers=4):
    """Create train and validation data loaders from data directory"""
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms
    val_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=transform_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, full_dataset.classes

def validate_model(model, val_loader, criterion, device):
    """Validate the model and return metrics"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_val_loss, accuracy, all_labels, all_preds

def train_classifier(
    classifier_type: str,
    data_dir: str,
    save_path: str,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = None,
    val_split: float = 0.2
):
    """Train a classifier for either 'tray' or 'dish'"""
    
    # Device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Training {classifier_type.upper()} classifier on {device}")
    print(f"{'='*50}")
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        data_dir, batch_size, val_split
    )
    
    print(f"Dataset classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Model
    model = models.efficientnet_b1(pretrained=True)
    num_classes = len(class_names)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss, val_acc, val_labels, val_preds = validate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = save_path.replace('.pt', '_best.pt')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_acc,
                'class_names': class_names
            }, best_model_path)
            print(f"  *** New best model saved! Accuracy: {val_acc:.4f}")
        
        print("-" * 30)
    
    # Final validation and classification report
    print(f"\nFinal Results for {classifier_type.upper()}:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    val_loss, val_acc, val_labels, val_preds = validate_model(
        model, val_loader, criterion, device
    )
    
    print(f"\nClassification Report for {classifier_type.upper()}:")
    print(classification_report(
        val_labels, val_preds, 
        target_names=class_names, 
        digits=4
    ))
    
    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'val_accuracy': val_acc,
        'class_names': class_names,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, save_path)
    
    print(f"\nFinal model saved to: {save_path}")
    print(f"Best model saved to: {best_model_path}")
    
    return {
        'final_accuracy': val_acc,
        'best_accuracy': best_val_acc,
        'class_names': class_names,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def main():
    parser = argparse.ArgumentParser(description='Train Tray and Dish Classifiers')
    parser.add_argument('--classifier', type=str, choices=['tray', 'dish', 'both'], 
                       default='both', help='Which classifier to train')
    parser.add_argument('--tray_data', type=str, default='AU/Dataset/Classification/tray',
                       help='Path to tray dataset')
    parser.add_argument('--dish_data', type=str, default='AU/Dataset/Classification/dish',
                       help='Path to dish dataset')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, 
                       help='Validation split ratio')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.classifier in ['tray', 'both']:
        print("Starting TRAY classifier training...")
        tray_results = train_classifier(
            classifier_type='tray',
            data_dir=args.tray_data,
            save_path=os.path.join(args.models_dir, 'tray_classifier.pt'),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            val_split=args.val_split
        )
        results['tray'] = tray_results
    
    if args.classifier in ['dish', 'both']:
        print("\nStarting DISH classifier training...")
        dish_results = train_classifier(
            classifier_type='dish',
            data_dir=args.dish_data,
            save_path=os.path.join(args.models_dir, 'dish_classifier.pt'),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            val_split=args.val_split
        )
        results['dish'] = dish_results
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for classifier_name, result in results.items():
        print(f"{classifier_name.upper()} Classifier:")
        print(f"  Classes: {result['class_names']}")
        print(f"  Final Accuracy: {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)")
        print(f"  Best Accuracy: {result['best_accuracy']:.4f} ({result['best_accuracy']*100:.2f}%)")
        print()

if __name__ == "__main__":
    # For direct execution without arguments
    import sys
    if len(sys.argv) == 1:
        # Default training for both classifiers
        results = {}
        
        print("Training both classifiers with default parameters...")
        
        # Train tray classifier
        tray_results = train_classifier(
            classifier_type='tray',
            data_dir='AU/Dataset/Classification/tray',
            save_path='models/tray_classifier.pt'
        )
        results['tray'] = tray_results
        
        # Train dish classifier
        dish_results = train_classifier(
            classifier_type='dish',
            data_dir='AU/Dataset/Classification/dish',
            save_path='models/dish_classifier.pt'
        )
        results['dish'] = dish_results
        
        # Summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        
        for classifier_name, result in results.items():
            print(f"{classifier_name.upper()} Classifier:")
            print(f"  Classes: {result['class_names']}")
            print(f"  Final Accuracy: {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)")
            print(f"  Best Accuracy: {result['best_accuracy']:.4f} ({result['best_accuracy']*100:.2f}%)")
            print()
    else:
        main()