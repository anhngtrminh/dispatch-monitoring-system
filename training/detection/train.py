#!/usr/bin/env python3
"""
YOLOv8 Training Script for Object Detection
Supports training, validation, and evaluation with comprehensive logging
"""

import os
import yaml
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json

class YOLOv8Trainer:
    def __init__(self, config_path="dataset.yaml"):
        """
        Initialize YOLOv8 trainer
        
        Args:
            config_path (str): Path to dataset configuration YAML file
        """
        self.config_path = config_path
        self.model = None
        self.results = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.output_dir = Path("runs") / f"detect_train_{self.timestamp}"
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate dataset configuration
        self._validate_dataset_config()
        
    def _validate_dataset_config(self):
        """Validate dataset configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Dataset config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
            
        required_keys = ['train', 'val', 'names']
        for key in required_keys:
            if key not in self.dataset_config:
                raise ValueError(f"Missing required key '{key}' in dataset config")
                
        print(f"Dataset configuration loaded successfully")
        print(f"Classes: {list(self.dataset_config['names'].values())}")
        print(f"Number of classes: {len(self.dataset_config['names'])}")
        
    def create_model(self, model_size='n', pretrained=True):
        """
        Create YOLOv8 model
        
        Args:
            model_size (str): Model size ('n', 's', 'm', 'l', 'x')
            pretrained (bool): Use pretrained weights
        """
        model_name = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
        self.model = YOLO(model_name)
        print(f"Model created: YOLOv8{model_size.upper()}")
        
    def train(self, epochs=20, imgsz=640, batch_size=64, **kwargs):
        """
        Train the YOLOv8 model
        
        Args:
            epochs (int): Number of training epochs
            imgsz (int): Image size for training
            batch_size (int): Batch size
            **kwargs: Additional training arguments
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
            
        # Default training arguments
        train_args = {
            'data': self.config_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'name': f'train_{self.timestamp}',
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'cache': False,  # Set to True if you have enough RAM
            'device': 'auto',  # Use GPU if available
            'workers': 8,
            'project': 'runs/detect',
            'patience': 50,  # Early stopping patience
            'optimizer': 'auto',  # SGD, Adam, AdamW, NAdam, RAdam, RMSProp
            'lr0': 0.01,  # Initial learning rate
            'lrf': 0.01,  # Final learning rate factor
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,  # Box loss gain
            'cls': 0.5,  # Class loss gain
            'dfl': 1.5,  # DFL loss gain
            'pose': 12.0,  # Pose loss gain
            'kobj': 2.0,  # Keypoint obj loss gain
            'label_smoothing': 0.0,
            'nbs': 64,  # Nominal batch size
            'hsv_h': 0.015,  # Image HSV-Hue augmentation
            'hsv_s': 0.7,  # Image HSV-Saturation augmentation
            'hsv_v': 0.4,  # Image HSV-Value augmentation
            'degrees': 0.0,  # Image rotation (+/- deg)
            'translate': 0.1,  # Image translation (+/- fraction)
            'scale': 0.5,  # Image scale (+/- gain)
            'shear': 0.0,  # Image shear (+/- deg)
            'perspective': 0.0,  # Image perspective (+/- fraction)
            'flipud': 0.0,  # Image flip up-down (probability)
            'fliplr': 0.5,  # Image flip left-right (probability)
            'mosaic': 1.0,  # Image mosaic (probability)
            'mixup': 0.0,  # Image mixup (probability)
            'copy_paste': 0.0,  # Segment copy-paste (probability)
        }
        
        # Update with any custom arguments
        train_args.update(kwargs)
        
        print(f"Starting training with the following configuration:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
            
        # Start training
        self.results = self.model.train(**train_args)
        
        print(f"Training completed!")
        print(f"Results saved to: {self.results.save_dir}")
        
        return self.results
        
    def validate(self, model_path=None, **kwargs):
        """
        Validate the trained model
        
        Args:
            model_path (str): Path to model weights (if None, uses last trained model)
            **kwargs: Additional validation arguments
        """
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for validation")
            
        val_args = {
            'data': self.config_path,
            'save_json': True,
            'save_hybrid': True,
            'conf': 0.001,
            'iou': 0.6,
            'max_det': 300,
            'half': False,
            'device': 'auto',
            'dnn': False,
            'plots': True,
            'rect': False,
            'split': 'val'
        }
        
        val_args.update(kwargs)
        
        print("Starting validation...")
        val_results = model.val(**val_args)
        
        print("Validation completed!")
        print(f"mAP50: {val_results.box.map50:.4f}")
        print(f"mAP50-95: {val_results.box.map:.4f}")
        
        return val_results
        
    def predict(self, source, model_path=None, save=True, **kwargs):
        """
        Run inference on images/videos
        
        Args:
            source (str): Path to image/video/directory
            model_path (str): Path to model weights
            save (bool): Save prediction results
            **kwargs: Additional prediction arguments
        """
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for prediction")
            
        pred_args = {
            'source': source,
            'conf': 0.25,
            'iou': 0.45,
            'half': False,
            'device': 'auto',
            'max_det': 1000,
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'embed': None,
            'show': False,
            'save': save,
            'save_frames': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'show_boxes': True,
            'line_width': None,
        }
        
        pred_args.update(kwargs)
        
        print(f"Running prediction on: {source}")
        results = model.predict(**pred_args)
        
        print("Prediction completed!")
        return results
        
    def export_model(self, model_path=None, format='onnx', **kwargs):
        """
        Export model to different formats
        
        Args:
            model_path (str): Path to model weights
            format (str): Export format ('onnx', 'torchscript', 'tflite', etc.)
            **kwargs: Additional export arguments
        """
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for export")
            
        export_args = {
            'format': format,
            'imgsz': 640,
            'keras': False,
            'optimize': False,
            'half': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': None,
            'workspace': 4,
            'nms': False,
        }
        
        export_args.update(kwargs)
        
        print(f"Exporting model to {format} format...")
        export_path = model.export(**export_args)
        
        print(f"Model exported to: {export_path}")
        return export_path
        
    def plot_training_results(self, results_path=None):
        """
        Plot training results
        
        Args:
            results_path (str): Path to results directory
        """
        if results_path is None and self.results:
            results_path = self.results.save_dir
        elif results_path is None:
            raise ValueError("No results available to plot")
            
        results_path = Path(results_path)
        
        # Load results CSV if available
        csv_path = results_path / 'results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()  # Remove whitespace
            
            # Create training plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Results', fontsize=16)
            
            # Loss plots
            if 'train/box_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
                axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
                axes[0, 0].set_title('Box Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            if 'train/cls_loss' in df.columns:
                axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
                axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
                axes[0, 1].set_title('Class Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # mAP plots
            if 'metrics/mAP50(B)' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
                axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
                axes[1, 0].set_title('mAP Metrics')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('mAP')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Precision/Recall
            if 'metrics/precision(B)' in df.columns:
                axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
                axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
                axes[1, 1].set_title('Precision & Recall')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plot_path = results_path / 'training_plots.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Training plots saved to: {plot_path}")
        else:
            print(f"Results CSV not found at: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Training Script')
    parser.add_argument('--config', type=str, default='training/detection/dataset.yaml', 
                       help='Path to dataset configuration file')
    parser.add_argument('--model-size', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=5, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Image size')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='Device to use (auto, cpu, 0, 1, etc.)')
    parser.add_argument('--workers', type=int, default=8, 
                       help='Number of worker threads')
    parser.add_argument('--patience', type=int, default=50, 
                       help='Early stopping patience')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--cache', action='store_true', 
                       help='Cache images for faster training')
    parser.add_argument('--validate', action='store_true', 
                       help='Run validation after training')
    parser.add_argument('--plot', action='store_true', 
                       help='Plot training results')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOv8Trainer(config_path=args.config)
    
    # Create model
    trainer.create_model(model_size=args.model_size, pretrained=args.pretrained)
    
    # Train model
    results = trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache
    )
    
    # Validate if requested
    if args.validate:
        trainer.validate()
    
    # Plot results if requested
    if args.plot:
        trainer.plot_training_results()
    
    print("\nTraining completed successfully!")
    print(f"Best model saved at: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Last model saved at: {results.save_dir / 'weights' / 'last.pt'}")

if __name__ == "__main__":
    main()