#!/usr/bin/env python3
"""
Simplified Malaysian IC Training Pipeline

Lightweight training pipeline for Malaysian Identity Card extraction that:
- Uses OCR + regex approach (no heavy ML dependencies)
- Integrates existing dataset with synthetic data generation
- Provides comprehensive evaluation metrics
- Focuses on rule-based optimization and validation

Author: AI Assistant
Date: 2025
"""

import os
import json
import yaml
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import random
import shutil
from PIL import Image
import time

# Import our custom modules
from malaysian_ic_synthetic_generator import MalaysianICGenerator
from malaysian_ic_extractor import MalaysianICExtractor
from malaysian_ic_validator import MalaysianICValidator

@dataclass
class SimpleTrainingConfig:
    """Simplified training configuration"""
    # Dataset paths
    dataset_root: str
    output_dir: str
    synthetic_data_dir: str
    
    # Training parameters
    num_synthetic_samples: int = 1000
    validation_split: float = 0.2
    
    # OCR optimization parameters
    test_ocr_configs: bool = True
    test_preprocessing: bool = True
    
    # Evaluation
    evaluation_metrics: List[str] = None
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                "field_accuracy", "character_error_rate", 
                "nric_validation_accuracy", "full_card_success_rate"
            ]

class SimpleMalaysianICTrainer:
    """Simplified trainer for Malaysian IC extraction"""
    
    def __init__(self, config: SimpleTrainingConfig):
        self.config = config
        
        # Create output directories first
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.synthetic_data_dir, exist_ok=True)
        
        # Setup logging after directories are created
        self.logger = self._setup_logging()
        
        # Initialize components
        self.synthetic_generator = MalaysianICGenerator()
        self.extractor = MalaysianICExtractor()
        self.validator = MalaysianICValidator()
        
        # Best configuration found during optimization
        self.best_config = {
            'ocr_config': None,
            'preprocessing_config': None,
            'performance_score': 0.0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler
        log_file = os.path.join(self.config.output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_existing_dataset(self) -> List[Dict]:
        """Load and process existing dataset"""
        self.logger.info("Loading existing dataset...")
        
        dataset_path = Path(self.config.dataset_root)
        data_yaml_path = dataset_path / "data.yaml"
        
        if not data_yaml_path.exists():
            self.logger.warning(f"data.yaml not found at {data_yaml_path}")
            return []
        
        # Load dataset configuration
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        dataset_items = []
        
        # Process train, val, test splits
        for split in ['train', 'valid', 'test']:
            split_path = dataset_path / split
            images_path = split_path / "images"
            
            if images_path.exists():
                for image_file in images_path.glob("*.jpg"):
                    # For now, we'll extract labels using OCR since we don't have ground truth
                    dataset_items.append({
                        'image_path': str(image_file),
                        'split': split,
                        'source': 'real'
                    })
        
        self.logger.info(f"Loaded {len(dataset_items)} items from existing dataset")
        return dataset_items
    
    def generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic training data"""
        self.logger.info(f"Generating {self.config.num_synthetic_samples} synthetic IC samples...")
        
        synthetic_items = []
        
        for i in range(self.config.num_synthetic_samples):
            try:
                # Generate synthetic IC data
                ic_data = self.synthetic_generator.generate_ic_data()
                
                # Create image
                filename = f"synthetic_ic_{i:06d}.png"
                image_path = os.path.join(self.config.synthetic_data_dir, filename)
                self.synthetic_generator.create_ic_image(ic_data, image_path)
                
                synthetic_items.append({
                    'image_path': image_path,
                    'ground_truth': ic_data,  # We know the ground truth for synthetic data
                    'split': 'synthetic',
                    'source': 'synthetic'
                })
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated {i + 1}/{self.config.num_synthetic_samples} synthetic samples")
            
            except Exception as e:
                self.logger.warning(f"Failed to generate synthetic sample {i}: {e}")
                continue
        
        self.logger.info(f"Successfully generated {len(synthetic_items)} synthetic samples")
        return synthetic_items
    
    def prepare_training_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Prepare training, validation, and test data"""
        self.logger.info("Preparing training data...")
        
        # Load existing dataset
        real_data = self.load_existing_dataset()
        
        # Generate synthetic data
        synthetic_data = self.generate_synthetic_data()
        
        # Split data
        train_data = []
        val_data = []
        test_data = []
        
        # Separate real data by existing splits
        real_train = [item for item in real_data if item['split'] == 'train']
        real_val = [item for item in real_data if item['split'] == 'valid']
        real_test = [item for item in real_data if item['split'] == 'test']
        
        # Split synthetic data
        if synthetic_data:
            random.shuffle(synthetic_data)
            val_size = int(len(synthetic_data) * self.config.validation_split)
            test_size = int(len(synthetic_data) * 0.1)  # 10% for test
            
            synth_val = synthetic_data[:val_size]
            synth_test = synthetic_data[val_size:val_size + test_size]
            synth_train = synthetic_data[val_size + test_size:]
        else:
            synth_train, synth_val, synth_test = [], [], []
        
        # Combine splits
        train_data = real_train + synth_train
        val_data = real_val + synth_val
        test_data = real_test + synth_test
        
        self.logger.info(f"Training data: {len(train_data)} samples ({len(real_train)} real, {len(synth_train)} synthetic)")
        self.logger.info(f"Validation data: {len(val_data)} samples ({len(real_val)} real, {len(synth_val)} synthetic)")
        self.logger.info(f"Test data: {len(test_data)} samples ({len(real_test)} real, {len(synth_test)} synthetic)")
        
        # Save data splits
        self._save_data_splits(train_data, val_data, test_data)
        
        return train_data, val_data, test_data
    
    def _save_data_splits(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """Save data splits to files"""
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, data in splits.items():
            split_file = os.path.join(self.config.output_dir, f"{split_name}_data.json")
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Saved {split_name} data to {split_file}")
    
    def optimize_ocr_configuration(self, val_data: List[Dict]) -> Dict:
        """Optimize OCR configuration using validation data"""
        self.logger.info("Optimizing OCR configuration...")
        
        # Test different OCR configurations
        ocr_configs = [
            {'lang': 'eng', 'psm': 6, 'oem': 3, 'name': 'eng_psm6'},
            {'lang': 'eng', 'psm': 8, 'oem': 3, 'name': 'eng_psm8'},
            {'lang': 'eng', 'psm': 11, 'oem': 3, 'name': 'eng_psm11'},
            {'lang': 'eng', 'psm': 13, 'oem': 3, 'name': 'eng_psm13'},
        ]
        
        # Test preprocessing configurations
        preprocessing_configs = [
            {'resize_factor': 2.0, 'denoise': True, 'sharpen': True, 'name': 'enhanced'},
            {'resize_factor': 1.5, 'denoise': False, 'sharpen': True, 'name': 'basic'},
            {'resize_factor': 3.0, 'denoise': True, 'sharpen': False, 'name': 'large'},
        ]
        
        best_score = 0.0
        best_ocr_config = None
        best_preprocessing_config = None
        
        # Use a subset of validation data for speed
        test_samples = val_data[:min(20, len(val_data))]
        
        for ocr_config in ocr_configs:
            for preprocessing_config in preprocessing_configs:
                self.logger.info(f"Testing OCR: {ocr_config['name']}, Preprocessing: {preprocessing_config['name']}")
                
                # Update extractor configuration
                self.extractor.tesseract_config = f"--psm {ocr_config['psm']} --oem {ocr_config['oem']}"
                self.extractor.resize_factor = preprocessing_config['resize_factor']
                self.extractor.use_denoise = preprocessing_config['denoise']
                self.extractor.use_sharpen = preprocessing_config['sharpen']
                
                # Evaluate configuration
                score = self._evaluate_configuration(test_samples)
                
                self.logger.info(f"Score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_ocr_config = ocr_config
                    best_preprocessing_config = preprocessing_config
        
        # Apply best configuration
        if best_ocr_config and best_preprocessing_config:
            self.extractor.tesseract_config = f"--psm {best_ocr_config['psm']} --oem {best_ocr_config['oem']}"
            self.extractor.resize_factor = best_preprocessing_config['resize_factor']
            self.extractor.use_denoise = best_preprocessing_config['denoise']
            self.extractor.use_sharpen = best_preprocessing_config['sharpen']
            
            self.best_config = {
                'ocr_config': best_ocr_config,
                'preprocessing_config': best_preprocessing_config,
                'performance_score': best_score
            }
        
        self.logger.info(f"Best configuration found - Score: {best_score:.3f}")
        self.logger.info(f"OCR: {best_ocr_config['name'] if best_ocr_config else 'None'}")
        self.logger.info(f"Preprocessing: {best_preprocessing_config['name'] if best_preprocessing_config else 'None'}")
        
        return self.best_config
    
    def _evaluate_configuration(self, test_samples: List[Dict]) -> float:
        """Evaluate a specific configuration"""
        total_score = 0.0
        valid_samples = 0
        
        for item in test_samples:
            try:
                # Extract fields
                extraction_result = self.extractor.extract_fields(item['image_path'])
                # Handle the ICExtractionResult object
                extracted = {}
                for field_name, field_obj in extraction_result.fields.items():
                    extracted[field_name] = field_obj.value
                
                # Calculate score based on validation and ground truth comparison
                validation_report = self.validator.validate_complete_ic(extracted)
                score = validation_report.confidence_score
                
                # If we have ground truth (synthetic data), compare accuracy
                if 'ground_truth' in item:
                    ground_truth = item['ground_truth']
                    accuracy_score = self._calculate_field_accuracy(extracted, ground_truth)
                    score = (score + accuracy_score) / 2  # Average validation and accuracy scores
                
                total_score += score
                valid_samples += 1
                
            except Exception as e:
                self.logger.warning(f"Evaluation failed for {item['image_path']}: {e}")
                continue
        
        return total_score / valid_samples if valid_samples > 0 else 0.0
    
    def _calculate_field_accuracy(self, extracted: Dict, ground_truth: Dict) -> float:
        """Calculate field-level accuracy between extracted and ground truth"""
        important_fields = ['nric', 'name', 'gender', 'birth_date', 'nationality']
        
        correct_fields = 0
        total_fields = 0
        
        for field in important_fields:
            if field in ground_truth and ground_truth[field]:
                total_fields += 1
                extracted_value = extracted.get(field, '').strip().upper()
                ground_truth_value = str(ground_truth[field]).strip().upper()
                
                if extracted_value == ground_truth_value:
                    correct_fields += 1
                elif field == 'nric':
                    # For NRIC, allow some formatting differences
                    extracted_clean = extracted_value.replace('-', '').replace(' ', '')
                    ground_truth_clean = ground_truth_value.replace('-', '').replace(' ', '')
                    if extracted_clean == ground_truth_clean:
                        correct_fields += 1
        
        return correct_fields / total_fields if total_fields > 0 else 0.0
    
    def train(self) -> Dict:
        """Main training function"""
        self.logger.info("Starting simplified Malaysian IC training pipeline...")
        
        start_time = datetime.now()
        
        # Prepare data
        train_data, val_data, test_data = self.prepare_training_data()
        
        # Optimize configuration using validation data
        if self.config.test_ocr_configs and val_data:
            optimization_results = self.optimize_ocr_configuration(val_data)
        else:
            optimization_results = {'performance_score': 0.0}
        
        # Evaluate on test data
        test_results = {}
        if test_data:
            self.logger.info("Evaluating on test data...")
            test_results = self.evaluate(test_data)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Compile results
        training_results = {
            'model_type': 'ocr_regex_optimized',
            'training_samples': len(train_data),
            'validation_samples': len(val_data),
            'test_samples': len(test_data),
            'training_time': training_time,
            'optimization_results': optimization_results,
            'test_results': test_results,
            'best_configuration': self.best_config
        }
        
        # Save results
        self._save_training_results(training_results)
        
        self.logger.info("Training pipeline completed successfully!")
        
        return training_results
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """Evaluate model on test data"""
        self.logger.info(f"Evaluating on {len(test_data)} test samples...")
        
        evaluation_results = {
            'total_samples': len(test_data),
            'field_accuracies': {},
            'validation_metrics': {},
            'full_card_success_rate': 0.0,
            'average_confidence': 0.0,
            'processing_time_per_sample': 0.0,
            'error_analysis': {}
        }
        
        field_correct = {}
        field_total = {}
        validation_correct = 0
        total_confidence = 0.0
        full_card_correct = 0
        total_processing_time = 0.0
        error_counts = {}
        
        for i, item in enumerate(test_data):
            try:
                start_time = time.time()
                
                # Extract fields
                extraction_result = self.extractor.extract_fields(item['image_path'])
                # Handle the ICExtractionResult object
                extracted = {}
                for field_name, field_obj in extraction_result.fields.items():
                    extracted[field_name] = field_obj.value
                
                # Validate
                validation_report = self.validator.validate_complete_ic(extracted)
                
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # Get ground truth if available
                ground_truth = item.get('ground_truth', {})
                
                # Calculate field accuracies
                for field_name in ['nric', 'name', 'gender', 'birth_date', 'religion', 'nationality', 'address']:
                    if field_name not in field_correct:
                        field_correct[field_name] = 0
                        field_total[field_name] = 0
                    
                    if field_name in ground_truth and ground_truth[field_name]:
                        field_total[field_name] += 1
                        
                        extracted_value = extracted.get(field_name, '').strip().upper()
                        ground_truth_value = str(ground_truth[field_name]).strip().upper()
                        
                        if extracted_value == ground_truth_value:
                            field_correct[field_name] += 1
                        elif field_name == 'nric':
                            # Allow formatting differences for NRIC
                            extracted_clean = extracted_value.replace('-', '').replace(' ', '')
                            ground_truth_clean = ground_truth_value.replace('-', '').replace(' ', '')
                            if extracted_clean == ground_truth_clean:
                                field_correct[field_name] += 1
                
                # Validation accuracy
                if validation_report.overall_valid:
                    validation_correct += 1
                
                # Count validation errors
                for field_result in validation_report.field_results.values():
                    for error in field_result.errors:
                        error_code = error.code
                        if error_code not in error_counts:
                            error_counts[error_code] = 0
                        error_counts[error_code] += 1
                
                # Full card success (all must-have fields correct)
                must_have_fields = ['nric', 'name', 'gender', 'birth_date', 'nationality']
                if ground_truth:
                    full_card_success = True
                    for field in must_have_fields:
                        if field in ground_truth and ground_truth[field]:
                            extracted_value = extracted.get(field, '').strip().upper()
                            ground_truth_value = str(ground_truth[field]).strip().upper()
                            if extracted_value != ground_truth_value:
                                full_card_success = False
                                break
                    
                    if full_card_success:
                        full_card_correct += 1
                
                # Confidence
                total_confidence += validation_report.confidence_score
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Evaluated {i + 1}/{len(test_data)} samples")
            
            except Exception as e:
                self.logger.warning(f"Evaluation failed for sample {i}: {e}")
                continue
        
        # Calculate final metrics
        for field_name in field_correct:
            if field_total[field_name] > 0:
                evaluation_results['field_accuracies'][field_name] = field_correct[field_name] / field_total[field_name]
        
        evaluation_results['validation_metrics']['overall_valid_rate'] = validation_correct / len(test_data)
        evaluation_results['full_card_success_rate'] = full_card_correct / len(test_data)
        evaluation_results['average_confidence'] = total_confidence / len(test_data)
        evaluation_results['processing_time_per_sample'] = total_processing_time / len(test_data)
        evaluation_results['error_analysis'] = error_counts
        
        self.logger.info("Evaluation completed!")
        self.logger.info(f"Full card success rate: {evaluation_results['full_card_success_rate']:.3f}")
        self.logger.info(f"Average confidence: {evaluation_results['average_confidence']:.3f}")
        self.logger.info(f"Processing time per sample: {evaluation_results['processing_time_per_sample']:.3f}s")
        
        return evaluation_results
    
    def _save_training_results(self, results: Dict):
        """Save training results to file"""
        results_file = os.path.join(self.config.output_dir, 'training_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Training results saved to {results_file}")
        
        # Also save configuration
        config_file = os.path.join(self.config.output_dir, 'training_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False, default=str)

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Malaysian IC extraction model (simplified)")
    parser.add_argument("--dataset_root", required=True, help="Path to dataset root directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for training results")
    parser.add_argument("--num_synthetic", type=int, default=1000, 
                       help="Number of synthetic samples to generate")
    parser.add_argument("--no_optimization", action="store_true", 
                       help="Skip OCR configuration optimization")
    
    args = parser.parse_args()
    
    # Create training configuration
    config = SimpleTrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        synthetic_data_dir=os.path.join(args.output_dir, "synthetic_data"),
        num_synthetic_samples=args.num_synthetic,
        test_ocr_configs=not args.no_optimization
    )
    
    # Initialize trainer
    trainer = SimpleMalaysianICTrainer(config)
    
    # Start training
    results = trainer.train()
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Results saved to: {args.output_dir}")
    print(f"🏆 Model type: OCR + Regex (Optimized)")
    
    if 'test_results' in results:
        test_results = results['test_results']
        print(f"📈 Full card success rate: {test_results['full_card_success_rate']:.3f}")
        print(f"🎯 Average confidence: {test_results['average_confidence']:.3f}")
        print(f"⏱️  Processing time per sample: {test_results['processing_time_per_sample']:.3f}s")
    
    if 'best_configuration' in results:
        best_config = results['best_configuration']
        print(f"🔧 Best performance score: {best_config['performance_score']:.3f}")

if __name__ == "__main__":
    main()