#!/usr/bin/env python3
"""
Enhanced Training Pipeline for Document Parser

Comprehensive training pipeline for document layout recognition and field detection
specifically designed for Malaysian IC and Passport documents.
"""

import logging
import json
import os
import time
import random
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data paths
    data_dir: str = "data/documents"
    annotations_dir: str = "data/annotations"
    output_dir: str = "models/enhanced_document_parser"
    
    # Model parameters
    model_type: str = "hybrid"  # 'cnn', 'transformer', 'hybrid'
    input_size: Tuple[int, int] = (512, 512)
    num_classes: int = 3  # IC, Passport, Other
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Augmentation parameters
    enable_augmentation: bool = True
    augmentation_probability: float = 0.8
    
    # Field detection parameters
    enable_field_detection: bool = True
    field_detection_threshold: float = 0.5
    
    # Training options
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # Monitoring
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    log_frequency: int = 100

@dataclass
class DocumentSample:
    """Document training sample"""
    image_path: str
    document_type: str
    fields: Dict[str, Any] = field(default_factory=dict)
    bboxes: Dict[str, List[int]] = field(default_factory=dict)  # [x1, y1, x2, y2]
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentDataset(Dataset):
    """Dataset for document classification and field detection"""
    
    def __init__(self, 
                 samples: List[DocumentSample],
                 config: TrainingConfig,
                 transform=None,
                 is_training: bool = True):
        self.samples = samples
        self.config = config
        self.transform = transform
        self.is_training = is_training
        
        # Document type to class mapping
        self.class_to_idx = {'ic': 0, 'passport': 1, 'other': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Field types for detection
        self.field_types = [
            'full_name', 'ic_number', 'passport_number', 'date_of_birth',
            'nationality', 'gender', 'issue_date', 'expiry_date', 'issuing_authority'
        ]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load image
            image = cv2.imread(sample.image_path)
            if image is None:
                raise ValueError(f"Could not load image: {sample.image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image.shape[:2]
            
            # Resize image
            image = cv2.resize(image, self.config.input_size)
            
            # Calculate scaling factors for bbox adjustment
            scale_x = self.config.input_size[0] / original_width
            scale_y = self.config.input_size[1] / original_height
            
            # Prepare field detection targets
            field_targets = self._prepare_field_targets(sample, scale_x, scale_y)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
            # Document classification target
            doc_class = self.class_to_idx.get(sample.document_type.lower(), 2)
            
            return {
                'image': image,
                'doc_class': torch.tensor(doc_class, dtype=torch.long),
                'field_targets': field_targets,
                'sample_path': sample.image_path,
                'metadata': sample.metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return dummy data
            dummy_image = torch.zeros(3, *self.config.input_size)
            dummy_targets = {field: torch.zeros(4) for field in self.field_types}
            
            return {
                'image': dummy_image,
                'doc_class': torch.tensor(2, dtype=torch.long),
                'field_targets': dummy_targets,
                'sample_path': sample.image_path,
                'metadata': {}
            }
    
    def _prepare_field_targets(self, sample: DocumentSample, scale_x: float, scale_y: float) -> Dict[str, torch.Tensor]:
        """Prepare field detection targets"""
        targets = {}
        
        for field_type in self.field_types:
            if field_type in sample.bboxes:
                bbox = sample.bboxes[field_type]
                # Scale bbox coordinates
                scaled_bbox = [
                    bbox[0] * scale_x,  # x1
                    bbox[1] * scale_y,  # y1
                    bbox[2] * scale_x,  # x2
                    bbox[3] * scale_y   # y2
                ]
                targets[field_type] = torch.tensor(scaled_bbox, dtype=torch.float32)
            else:
                # No field present
                targets[field_type] = torch.zeros(4, dtype=torch.float32)
        
        return targets

class DocumentClassificationModel(nn.Module):
    """Document classification model"""
    
    def __init__(self, num_classes: int = 3, model_type: str = "hybrid"):
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        
        if model_type == "cnn":
            self._build_cnn_model()
        elif model_type == "transformer":
            self._build_transformer_model()
        else:  # hybrid
            self._build_hybrid_model()
    
    def _build_cnn_model(self):
        """Build CNN-based model"""
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes)
        )
    
    def _build_transformer_model(self):
        """Build transformer-based model"""
        # Simplified transformer for document classification
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, 768))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, self.num_classes)
        )
    
    def _build_hybrid_model(self):
        """Build hybrid CNN-Transformer model"""
        # CNN feature extractor
        self.cnn_features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Transformer for spatial relationships
        self.spatial_embed = nn.Linear(256, 512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, x):
        if self.model_type == "cnn":
            features = self.features(x)
            output = self.classifier(features)
        elif self.model_type == "transformer":
            # Patch embedding
            x = self.patch_embed(x)  # [B, 768, H/16, W/16]
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, 768]
            
            # Add positional embedding
            x = x + self.pos_embed[:, :x.size(1), :]
            
            # Transformer
            x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
            
            # Classification
            x = x.mean(dim=1)  # Global average pooling
            output = self.classifier(x)
        else:  # hybrid
            # CNN features
            cnn_features = self.cnn_features(x)  # [B, 256, H/8, W/8]
            B, C, H, W = cnn_features.shape
            
            # Reshape for transformer
            features = cnn_features.flatten(2).transpose(1, 2)  # [B, H*W, 256]
            features = self.spatial_embed(features)  # [B, H*W, 512]
            
            # Transformer
            features = self.transformer(features.transpose(0, 1)).transpose(0, 1)
            
            # Classification
            output = self.classifier(features.transpose(1, 2))
        
        return output

class FieldDetectionModel(nn.Module):
    """Field detection model for bounding box regression"""
    
    def __init__(self, num_fields: int = 9, backbone_type: str = "resnet"):
        super().__init__()
        self.num_fields = num_fields
        
        # Backbone
        if backbone_type == "resnet":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                # ResNet blocks (simplified)
                self._make_layer(64, 128, 2),
                self._make_layer(128, 256, 2),
                self._make_layer(256, 512, 2),
            )
        
        # Field detection heads
        self.field_heads = nn.ModuleDict()
        for i in range(num_fields):
            self.field_heads[f'field_{i}'] = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 4)  # x1, y1, x2, y2
            )
    
    def _make_layer(self, in_channels, out_channels, stride):
        """Create a simple residual layer"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        outputs = {}
        for field_name, head in self.field_heads.items():
            outputs[field_name] = head(features)
        
        return outputs

class EnhancedTrainingPipeline:
    """Enhanced training pipeline for document parser"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'logs'), exist_ok=True)
        
        # Initialize models
        self.classification_model = None
        self.field_detection_model = None
        
        # Training state
        self.training_history = {
            'classification': {'train_loss': [], 'val_loss': [], 'val_accuracy': []},
            'field_detection': {'train_loss': [], 'val_loss': [], 'val_iou': []}
        }
        
        logger.info(f"Training pipeline initialized on device: {self.device}")
    
    def prepare_data(self, data_dir: str) -> Tuple[List[DocumentSample], List[DocumentSample], List[DocumentSample]]:
        """
        Prepare training data from directory structure.
        
        Expected structure:
        data_dir/
        ├── ic/
        │   ├── images/
        │   └── annotations/
        ├── passport/
        │   ├── images/
        │   └── annotations/
        └── other/
            ├── images/
            └── annotations/
        """
        samples = []
        
        try:
            for doc_type in ['ic', 'passport', 'other']:
                doc_dir = os.path.join(data_dir, doc_type)
                images_dir = os.path.join(doc_dir, 'images')
                annotations_dir = os.path.join(doc_dir, 'annotations')
                
                if not os.path.exists(images_dir):
                    logger.warning(f"Images directory not found: {images_dir}")
                    continue
                
                # Load samples
                for image_file in os.listdir(images_dir):
                    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    image_path = os.path.join(images_dir, image_file)
                    annotation_file = os.path.splitext(image_file)[0] + '.json'
                    annotation_path = os.path.join(annotations_dir, annotation_file)
                    
                    # Load annotation if exists
                    fields = {}
                    bboxes = {}
                    if os.path.exists(annotation_path):
                        try:
                            with open(annotation_path, 'r') as f:
                                annotation = json.load(f)
                                fields = annotation.get('fields', {})
                                bboxes = annotation.get('bboxes', {})
                        except Exception as e:
                            logger.warning(f"Failed to load annotation {annotation_path}: {e}")
                    
                    sample = DocumentSample(
                        image_path=image_path,
                        document_type=doc_type,
                        fields=fields,
                        bboxes=bboxes,
                        metadata={'source': 'training_data'}
                    )
                    samples.append(sample)
            
            logger.info(f"Loaded {len(samples)} samples")
            
            # Split data
            train_samples, temp_samples = train_test_split(
                samples, test_size=self.config.validation_split + self.config.test_split, 
                random_state=42, stratify=[s.document_type for s in samples]
            )
            
            val_samples, test_samples = train_test_split(
                temp_samples, 
                test_size=self.config.test_split / (self.config.validation_split + self.config.test_split),
                random_state=42, stratify=[s.document_type for s in temp_samples]
            )
            
            logger.info(f"Data split - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
            
            return train_samples, val_samples, test_samples
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return [], [], []
    
    def create_data_augmentation(self) -> A.Compose:
        """Create data augmentation pipeline"""
        if not self.config.enable_augmentation:
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        return A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            ], p=0.7),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=7, p=0.2),
            ], p=0.4),
            
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
            ], p=0.6),
            
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
            ], p=0.3),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], p=self.config.augmentation_probability)
    
    def train_classification_model(self, train_samples: List[DocumentSample], val_samples: List[DocumentSample]) -> Dict[str, Any]:
        """Train document classification model"""
        logger.info("Starting classification model training...")
        
        try:
            # Create datasets
            train_transform = self.create_data_augmentation()
            val_transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            train_dataset = DocumentDataset(train_samples, self.config, train_transform, is_training=True)
            val_dataset = DocumentDataset(val_samples, self.config, val_transform, is_training=False)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
            
            # Initialize model
            self.classification_model = DocumentClassificationModel(
                num_classes=self.config.num_classes,
                model_type=self.config.model_type
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                self.classification_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            
            # Training loop
            best_val_accuracy = 0.0
            patience_counter = 0
            
            for epoch in range(self.config.num_epochs):
                # Training phase
                self.classification_model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    images = batch['image'].to(self.device)
                    labels = batch['doc_class'].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.classification_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                    
                    if batch_idx % self.config.log_frequency == 0:
                        logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                # Validation phase
                self.classification_model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch['image'].to(self.device)
                        labels = batch['doc_class'].to(self.device)
                        
                        outputs = self.classification_model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                
                # Calculate metrics
                train_accuracy = 100.0 * train_correct / train_total
                val_accuracy = 100.0 * val_correct / val_total
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                # Update history
                self.training_history['classification']['train_loss'].append(avg_train_loss)
                self.training_history['classification']['val_loss'].append(avg_val_loss)
                self.training_history['classification']['val_accuracy'].append(val_accuracy)
                
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                           f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                
                # Early stopping and checkpointing
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.classification_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': val_accuracy,
                        'config': self.config
                    }, os.path.join(self.config.output_dir, 'best_classification_model.pth'))
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Save checkpoint
                if self.config.save_checkpoints and epoch % self.config.checkpoint_frequency == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.classification_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': val_accuracy,
                        'config': self.config
                    }, os.path.join(self.config.output_dir, 'checkpoints', f'classification_epoch_{epoch}.pth'))
                
                scheduler.step()
            
            return {
                'best_val_accuracy': best_val_accuracy,
                'training_history': self.training_history['classification'],
                'model_path': os.path.join(self.config.output_dir, 'best_classification_model.pth')
            }
            
        except Exception as e:
            logger.error(f"Classification training failed: {e}")
            return {'error': str(e)}
    
    def train_field_detection_model(self, train_samples: List[DocumentSample], val_samples: List[DocumentSample]) -> Dict[str, Any]:
        """Train field detection model"""
        if not self.config.enable_field_detection:
            return {'message': 'Field detection training disabled'}
        
        logger.info("Starting field detection model training...")
        
        try:
            # Filter samples with field annotations
            train_samples_with_fields = [s for s in train_samples if s.bboxes]
            val_samples_with_fields = [s for s in val_samples if s.bboxes]
            
            if not train_samples_with_fields:
                logger.warning("No training samples with field annotations found")
                return {'error': 'No field annotations available'}
            
            # Create datasets
            train_transform = self.create_data_augmentation()
            val_transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            train_dataset = DocumentDataset(train_samples_with_fields, self.config, train_transform, is_training=True)
            val_dataset = DocumentDataset(val_samples_with_fields, self.config, val_transform, is_training=False)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
            
            # Initialize model
            self.field_detection_model = FieldDetectionModel(
                num_fields=len(train_dataset.field_types)
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.SmoothL1Loss()  # Better for bounding box regression
            optimizer = optim.AdamW(
                self.field_detection_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            
            # Training loop
            best_val_iou = 0.0
            patience_counter = 0
            
            for epoch in range(self.config.num_epochs):
                # Training phase
                self.field_detection_model.train()
                train_loss = 0.0
                
                for batch_idx, batch in enumerate(train_loader):
                    images = batch['image'].to(self.device)
                    field_targets = batch['field_targets']
                    
                    optimizer.zero_grad()
                    outputs = self.field_detection_model(images)
                    
                    # Calculate loss for each field
                    total_loss = 0.0
                    for field_idx, field_name in enumerate(train_dataset.field_types):
                        if f'field_{field_idx}' in outputs:
                            target = field_targets[field_name].to(self.device)
                            pred = outputs[f'field_{field_idx}']
                            
                            # Only calculate loss for valid targets (non-zero)
                            valid_mask = target.sum(dim=1) > 0
                            if valid_mask.any():
                                field_loss = criterion(pred[valid_mask], target[valid_mask])
                                total_loss += field_loss
                    
                    if total_loss > 0:
                        total_loss.backward()
                        optimizer.step()
                        train_loss += total_loss.item()
                    
                    if batch_idx % self.config.log_frequency == 0:
                        logger.info(f"Field Detection Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")
                
                # Validation phase
                self.field_detection_model.eval()
                val_loss = 0.0
                val_iou_scores = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch['image'].to(self.device)
                        field_targets = batch['field_targets']
                        
                        outputs = self.field_detection_model(images)
                        
                        # Calculate validation loss and IoU
                        batch_loss = 0.0
                        batch_ious = []
                        
                        for field_idx, field_name in enumerate(train_dataset.field_types):
                            if f'field_{field_idx}' in outputs:
                                target = field_targets[field_name].to(self.device)
                                pred = outputs[f'field_{field_idx}']
                                
                                valid_mask = target.sum(dim=1) > 0
                                if valid_mask.any():
                                    field_loss = criterion(pred[valid_mask], target[valid_mask])
                                    batch_loss += field_loss
                                    
                                    # Calculate IoU for valid predictions
                                    iou = self._calculate_iou(pred[valid_mask], target[valid_mask])
                                    batch_ious.extend(iou.cpu().numpy())
                        
                        val_loss += batch_loss.item()
                        val_iou_scores.extend(batch_ious)
                
                # Calculate metrics
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
                avg_val_iou = np.mean(val_iou_scores) if val_iou_scores else 0
                
                # Update history
                self.training_history['field_detection']['train_loss'].append(avg_train_loss)
                self.training_history['field_detection']['val_loss'].append(avg_val_loss)
                self.training_history['field_detection']['val_iou'].append(avg_val_iou)
                
                logger.info(f"Field Detection Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")
                
                # Early stopping and checkpointing
                if avg_val_iou > best_val_iou:
                    best_val_iou = avg_val_iou
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.field_detection_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_iou': avg_val_iou,
                        'config': self.config
                    }, os.path.join(self.config.output_dir, 'best_field_detection_model.pth'))
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                scheduler.step()
            
            return {
                'best_val_iou': best_val_iou,
                'training_history': self.training_history['field_detection'],
                'model_path': os.path.join(self.config.output_dir, 'best_field_detection_model.pth')
            }
            
        except Exception as e:
            logger.error(f"Field detection training failed: {e}")
            return {'error': str(e)}
    
    def _calculate_iou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between predicted and target bounding boxes"""
        # pred_boxes and target_boxes: [N, 4] (x1, y1, x2, y2)
        
        # Calculate intersection
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        # Calculate union
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def evaluate_models(self, test_samples: List[DocumentSample]) -> Dict[str, Any]:
        """Evaluate trained models on test set"""
        logger.info("Evaluating models on test set...")
        
        results = {}
        
        try:
            # Evaluate classification model
            if self.classification_model is not None:
                classification_results = self._evaluate_classification(test_samples)
                results['classification'] = classification_results
            
            # Evaluate field detection model
            if self.field_detection_model is not None and self.config.enable_field_detection:
                field_detection_results = self._evaluate_field_detection(test_samples)
                results['field_detection'] = field_detection_results
            
            return results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_classification(self, test_samples: List[DocumentSample]) -> Dict[str, Any]:
        """Evaluate classification model"""
        test_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        test_dataset = DocumentDataset(test_samples, self.config, test_transform, is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        
        self.classification_model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['doc_class'].to(self.device)
                
                outputs = self.classification_model(images)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Generate classification report
        class_names = ['IC', 'Passport', 'Other']
        report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
        confusion_mat = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_mat.tolist(),
            'class_names': class_names
        }
    
    def _evaluate_field_detection(self, test_samples: List[DocumentSample]) -> Dict[str, Any]:
        """Evaluate field detection model"""
        test_samples_with_fields = [s for s in test_samples if s.bboxes]
        
        if not test_samples_with_fields:
            return {'error': 'No test samples with field annotations'}
        
        test_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        test_dataset = DocumentDataset(test_samples_with_fields, self.config, test_transform, is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        
        self.field_detection_model.eval()
        field_ious = {field: [] for field in test_dataset.field_types}
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                field_targets = batch['field_targets']
                
                outputs = self.field_detection_model(images)
                
                for field_idx, field_name in enumerate(test_dataset.field_types):
                    if f'field_{field_idx}' in outputs:
                        target = field_targets[field_name].to(self.device)
                        pred = outputs[f'field_{field_idx}']
                        
                        valid_mask = target.sum(dim=1) > 0
                        if valid_mask.any():
                            iou = self._calculate_iou(pred[valid_mask], target[valid_mask])
                            field_ious[field_name].extend(iou.cpu().numpy())
        
        # Calculate average IoU per field
        field_metrics = {}
        overall_ious = []
        
        for field_name, ious in field_ious.items():
            if ious:
                avg_iou = np.mean(ious)
                field_metrics[field_name] = {
                    'avg_iou': avg_iou,
                    'num_samples': len(ious),
                    'precision_at_50': np.mean(np.array(ious) > 0.5),
                    'precision_at_75': np.mean(np.array(ious) > 0.75)
                }
                overall_ious.extend(ious)
        
        overall_avg_iou = np.mean(overall_ious) if overall_ious else 0
        
        return {
            'overall_avg_iou': overall_avg_iou,
            'field_metrics': field_metrics,
            'total_samples': len(test_samples_with_fields)
        }
    
    def save_training_report(self) -> str:
        """Save comprehensive training report"""
        try:
            report = {
                'training_config': {
                    'model_type': self.config.model_type,
                    'input_size': self.config.input_size,
                    'batch_size': self.config.batch_size,
                    'learning_rate': self.config.learning_rate,
                    'num_epochs': self.config.num_epochs,
                    'augmentation_enabled': self.config.enable_augmentation
                },
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat(),
                'device_used': str(self.device)
            }
            
            report_path = os.path.join(self.config.output_dir, 'training_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Training report saved to: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to save training report: {e}")
            return ""
    
    def run_full_training_pipeline(self, data_dir: str) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        try:
            # Prepare data
            train_samples, val_samples, test_samples = self.prepare_data(data_dir)
            
            if not train_samples:
                return {'error': 'No training data found'}
            
            results = {
                'data_summary': {
                    'train_samples': len(train_samples),
                    'val_samples': len(val_samples),
                    'test_samples': len(test_samples)
                }
            }
            
            # Train classification model
            classification_results = self.train_classification_model(train_samples, val_samples)
            results['classification_training'] = classification_results
            
            # Train field detection model
            if self.config.enable_field_detection:
                field_detection_results = self.train_field_detection_model(train_samples, val_samples)
                results['field_detection_training'] = field_detection_results
            
            # Evaluate models
            if test_samples:
                evaluation_results = self.evaluate_models(test_samples)
                results['evaluation'] = evaluation_results
            
            # Save training report
            report_path = self.save_training_report()
            results['report_path'] = report_path
            
            logger.info("Training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {'error': str(e)}

# Example usage and configuration
def create_default_config() -> TrainingConfig:
    """Create default training configuration"""
    return TrainingConfig(
        data_dir="data/documents",
        output_dir="models/enhanced_document_parser",
        model_type="hybrid",
        batch_size=16,
        learning_rate=0.001,
        num_epochs=50,
        enable_augmentation=True,
        enable_field_detection=True
    )

if __name__ == "__main__":
    # Example training script
    config = create_default_config()
    pipeline = EnhancedTrainingPipeline(config)
    
    # Run training
    results = pipeline.run_full_training_pipeline(config.data_dir)
    
    if 'error' in results:
        logger.error(f"Training failed: {results['error']}")
    else:
        logger.info("Training completed successfully!")
        logger.info(f"Results: {json.dumps(results, indent=2)}")