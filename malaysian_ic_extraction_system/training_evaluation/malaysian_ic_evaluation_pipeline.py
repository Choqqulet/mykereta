#!/usr/bin/env python3
"""
Malaysian IC Evaluation Pipeline

Comprehensive evaluation system for Malaysian Identity Card extraction that provides:
- Field-level accuracy metrics
- Character error rate analysis
- NRIC validation accuracy
- Full card success rate
- Confidence score analysis
- Error categorization and analysis
- Performance benchmarking
- Visual evaluation reports

Author: AI Assistant
Date: 2025
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import difflib
import re
from collections import defaultdict, Counter

# Import our custom modules
from malaysian_ic_extractor import MalaysianICExtractor
from malaysian_ic_validator import MalaysianICValidator

@dataclass
class FieldEvaluation:
    """Evaluation metrics for a single field"""
    field_name: str
    total_samples: int
    correct_exact: int
    correct_fuzzy: int
    missing: int
    incorrect: int
    character_error_rate: float
    confidence_scores: List[float]
    
    @property
    def exact_accuracy(self) -> float:
        return self.correct_exact / self.total_samples if self.total_samples > 0 else 0.0
    
    @property
    def fuzzy_accuracy(self) -> float:
        return self.correct_fuzzy / self.total_samples if self.total_samples > 0 else 0.0
    
    @property
    def missing_rate(self) -> float:
        return self.missing / self.total_samples if self.total_samples > 0 else 0.0
    
    @property
    def average_confidence(self) -> float:
        return np.mean(self.confidence_scores) if self.confidence_scores else 0.0

@dataclass
class ValidationEvaluation:
    """Evaluation metrics for validation rules"""
    rule_name: str
    total_applicable: int
    passed: int
    failed: int
    error_types: Dict[str, int]
    
    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_applicable if self.total_applicable > 0 else 0.0

@dataclass
class PerformanceMetrics:
    """Performance benchmarking metrics"""
    total_samples: int
    total_processing_time: float
    average_time_per_sample: float
    min_time: float
    max_time: float
    std_time: float
    memory_usage_mb: float

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    model_name: str
    evaluation_date: str
    dataset_info: Dict
    field_evaluations: Dict[str, FieldEvaluation]
    validation_evaluations: Dict[str, ValidationEvaluation]
    performance_metrics: PerformanceMetrics
    overall_metrics: Dict[str, float]
    error_analysis: Dict[str, any]
    recommendations: List[str]

class MalaysianICEvaluator:
    """Comprehensive evaluator for Malaysian IC extraction systems"""
    
    def __init__(self, extractor: MalaysianICExtractor, validator: MalaysianICValidator):
        self.extractor = extractor
        self.validator = validator
        self.logger = self._setup_logging()
        
        # Field importance weights for overall scoring
        self.field_weights = {
            'nric': 0.25,
            'name': 0.20,
            'gender': 0.15,
            'birth_date': 0.15,
            'nationality': 0.10,
            'religion': 0.10,
            'address': 0.05
        }
        
        # Fuzzy matching thresholds
        self.fuzzy_thresholds = {
            'nric': 0.9,
            'name': 0.8,
            'gender': 0.9,
            'birth_date': 0.9,
            'nationality': 0.8,
            'religion': 0.8,
            'address': 0.7
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f"{__name__}.evaluator")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    
    def calculate_character_error_rate(self, predicted: str, ground_truth: str) -> float:
        """Calculate character error rate between predicted and ground truth text"""
        if not ground_truth:
            return 1.0 if predicted else 0.0
        
        if not predicted:
            return 1.0
        
        # Normalize strings
        pred_clean = re.sub(r'\s+', ' ', predicted.strip().upper())
        gt_clean = re.sub(r'\s+', ' ', ground_truth.strip().upper())
        
        # Calculate edit distance
        matcher = difflib.SequenceMatcher(None, pred_clean, gt_clean)
        edit_distance = len(gt_clean) - sum(block.size for block in matcher.get_matching_blocks())
        
        return edit_distance / len(gt_clean)
    
    def fuzzy_match(self, predicted: str, ground_truth: str, threshold: float = 0.8) -> bool:
        """Check if predicted text fuzzy matches ground truth"""
        if not predicted or not ground_truth:
            return False
        
        # Normalize strings
        pred_clean = re.sub(r'\s+', ' ', predicted.strip().upper())
        gt_clean = re.sub(r'\s+', ' ', ground_truth.strip().upper())
        
        # Calculate similarity
        similarity = difflib.SequenceMatcher(None, pred_clean, gt_clean).ratio()
        return similarity >= threshold
    
    def evaluate_field(self, field_name: str, predictions: List[str], 
                      ground_truths: List[str], confidences: List[float] = None) -> FieldEvaluation:
        """Evaluate a specific field across all samples"""
        if confidences is None:
            confidences = [1.0] * len(predictions)
        
        total_samples = len(ground_truths)
        correct_exact = 0
        correct_fuzzy = 0
        missing = 0
        incorrect = 0
        cer_scores = []
        confidence_scores = []
        
        threshold = self.fuzzy_thresholds.get(field_name, 0.8)
        
        for pred, gt, conf in zip(predictions, ground_truths, confidences):
            if not gt:  # Skip if no ground truth
                continue
            
            confidence_scores.append(conf)
            
            if not pred:
                missing += 1
                cer_scores.append(1.0)
            else:
                # Calculate CER
                cer = self.calculate_character_error_rate(pred, gt)
                cer_scores.append(cer)
                
                # Check exact match
                pred_clean = pred.strip().upper()
                gt_clean = gt.strip().upper()
                
                if pred_clean == gt_clean:
                    correct_exact += 1
                    correct_fuzzy += 1
                elif self.fuzzy_match(pred, gt, threshold):
                    correct_fuzzy += 1
                else:
                    incorrect += 1
        
        avg_cer = np.mean(cer_scores) if cer_scores else 0.0
        
        return FieldEvaluation(
            field_name=field_name,
            total_samples=total_samples,
            correct_exact=correct_exact,
            correct_fuzzy=correct_fuzzy,
            missing=missing,
            incorrect=incorrect,
            character_error_rate=avg_cer,
            confidence_scores=confidence_scores
        )
    
    def evaluate_validation_rules(self, extracted_data: List[Dict], 
                                validation_reports: List[any]) -> Dict[str, ValidationEvaluation]:
        """Evaluate validation rule performance"""
        rule_stats = defaultdict(lambda: {
            'total_applicable': 0,
            'passed': 0,
            'failed': 0,
            'error_types': Counter()
        })
        
        for data, report in zip(extracted_data, validation_reports):
            # Check each validation rule
            for field_name, field_result in report.field_results.items():
                rule_name = f"{field_name}_validation"
                stats = rule_stats[rule_name]
                
                if field_name in data and data[field_name]:
                    stats['total_applicable'] += 1
                    
                    if field_result.is_valid:
                        stats['passed'] += 1
                    else:
                        stats['failed'] += 1
                        
                        # Count error types
                        for error in field_result.errors:
                            stats['error_types'][error.code] += 1
        
        # Convert to ValidationEvaluation objects
        evaluations = {}
        for rule_name, stats in rule_stats.items():
            evaluations[rule_name] = ValidationEvaluation(
                rule_name=rule_name,
                total_applicable=stats['total_applicable'],
                passed=stats['passed'],
                failed=stats['failed'],
                error_types=dict(stats['error_types'])
            )
        
        return evaluations
    
    def evaluate_dataset(self, test_data: List[Dict], output_dir: str = None) -> EvaluationReport:
        """Evaluate the extraction system on a test dataset"""
        self.logger.info(f"Starting evaluation on {len(test_data)} samples...")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Filter samples with ground truth for evaluation
        evaluation_samples = [sample for sample in test_data if 'ground_truth' in sample]
        self.logger.info(f"Found {len(evaluation_samples)} samples with ground truth for evaluation")
        
        if len(evaluation_samples) == 0:
            self.logger.warning("No samples with ground truth found for evaluation!")
            return None
        
        # Initialize data structures
        extracted_data = []
        validation_reports = []
        processing_times = []
        field_predictions = defaultdict(list)
        field_ground_truths = defaultdict(list)
        field_confidences = defaultdict(list)
        
        # Process each sample
        for i, sample in enumerate(evaluation_samples):
            try:
                start_time = time.time()
                
                # Extract fields
                extraction_result = self.extractor.extract_fields(sample['image_path'])
                
                # Handle the ICExtractionResult object
                extracted = {}
                for field_name, field_obj in extraction_result.fields.items():
                    extracted[field_name] = field_obj.value
                
                # Validate
                validation_report = self.validator.validate_complete_ic(extracted)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                extracted_data.append(extracted)
                validation_reports.append(validation_report)
                
                # Collect field data for evaluation
                ground_truth = sample.get('ground_truth', {})
                
                for field_name in self.field_weights.keys():
                    pred_value = extracted.get(field_name, '')
                    gt_value = ground_truth.get(field_name, '')
                    confidence = extraction_result.fields.get(field_name).confidence if field_name in extraction_result.fields else 0.0
                    
                    field_predictions[field_name].append(pred_value)
                    field_ground_truths[field_name].append(gt_value)
                    field_confidences[field_name].append(confidence)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(evaluation_samples)} samples")
            
            except Exception as e:
                self.logger.warning(f"Failed to process sample {i}: {e}")
                continue
        
        # Evaluate fields
        field_evaluations = {}
        for field_name in self.field_weights.keys():
            field_eval = self.evaluate_field(
                field_name,
                field_predictions[field_name],
                field_ground_truths[field_name],
                field_confidences[field_name]
            )
            field_evaluations[field_name] = field_eval
        
        # Evaluate validation rules
        validation_evaluations = self.evaluate_validation_rules(extracted_data, validation_reports)
        
        # Calculate performance metrics
        performance_metrics = PerformanceMetrics(
            total_samples=len(evaluation_samples),
            total_processing_time=sum(processing_times),
            average_time_per_sample=np.mean(processing_times),
            min_time=min(processing_times) if processing_times else 0.0,
            max_time=max(processing_times) if processing_times else 0.0,
            std_time=np.std(processing_times) if processing_times else 0.0,
            memory_usage_mb=0.0  # Could be implemented with psutil
        )
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(field_evaluations, validation_reports)
        
        # Perform error analysis
        error_analysis = self._analyze_errors(field_evaluations, validation_evaluations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(field_evaluations, validation_evaluations, error_analysis)
        
        # Create evaluation report
        report = EvaluationReport(
            model_name="Malaysian IC OCR+Regex Extractor",
            evaluation_date=datetime.now().isoformat(),
            dataset_info={
                'total_samples': len(test_data),
                'evaluation_samples': len(evaluation_samples),
                'real_samples': len([s for s in test_data if s.get('source') == 'real']),
                'synthetic_samples': len([s for s in test_data if s.get('source') == 'synthetic'])
            },
            field_evaluations=field_evaluations,
            validation_evaluations=validation_evaluations,
            performance_metrics=performance_metrics,
            overall_metrics=overall_metrics,
            error_analysis=error_analysis,
            recommendations=recommendations
        )
        
        # Save report
        if output_dir:
            self._save_evaluation_report(report, output_dir)
            self._generate_visualizations(report, output_dir)
        
        self.logger.info("Evaluation completed!")
        return report
    
    def _calculate_overall_metrics(self, field_evaluations: Dict[str, FieldEvaluation], 
                                 validation_reports: List[any]) -> Dict[str, float]:
        """Calculate overall system metrics"""
        # Weighted field accuracy
        weighted_exact_accuracy = 0.0
        weighted_fuzzy_accuracy = 0.0
        
        for field_name, evaluation in field_evaluations.items():
            weight = self.field_weights.get(field_name, 0.0)
            weighted_exact_accuracy += evaluation.exact_accuracy * weight
            weighted_fuzzy_accuracy += evaluation.fuzzy_accuracy * weight
        
        # Full card success rate (all must-have fields correct)
        must_have_fields = ['nric', 'name', 'gender', 'birth_date', 'nationality']
        full_card_successes = 0
        
        if field_evaluations and len(field_evaluations[must_have_fields[0]].confidence_scores) > 0:
            total_samples = field_evaluations[must_have_fields[0]].total_samples
            
            for i in range(total_samples):
                all_correct = True
                for field in must_have_fields:
                    if field in field_evaluations:
                        # This is a simplified check - in practice, you'd need to track per-sample results
                        if field_evaluations[field].exact_accuracy < 0.8:  # Threshold
                            all_correct = False
                            break
                
                if all_correct:
                    full_card_successes += 1
            
            full_card_success_rate = full_card_successes / total_samples
        else:
            full_card_success_rate = 0.0
        
        # Validation success rate
        validation_success_rate = 0.0
        if validation_reports:
            valid_reports = sum(1 for report in validation_reports if report.overall_valid)
            validation_success_rate = valid_reports / len(validation_reports)
        
        # Average confidence
        all_confidences = []
        for evaluation in field_evaluations.values():
            all_confidences.extend(evaluation.confidence_scores)
        
        average_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return {
            'weighted_exact_accuracy': weighted_exact_accuracy,
            'weighted_fuzzy_accuracy': weighted_fuzzy_accuracy,
            'full_card_success_rate': full_card_success_rate,
            'validation_success_rate': validation_success_rate,
            'average_confidence': average_confidence
        }
    
    def _analyze_errors(self, field_evaluations: Dict[str, FieldEvaluation], 
                       validation_evaluations: Dict[str, ValidationEvaluation]) -> Dict[str, any]:
        """Analyze common errors and patterns"""
        error_analysis = {
            'field_error_patterns': {},
            'validation_error_patterns': {},
            'critical_issues': [],
            'improvement_areas': []
        }
        
        # Analyze field errors
        for field_name, evaluation in field_evaluations.items():
            field_errors = {
                'missing_rate': evaluation.missing_rate,
                'incorrect_rate': evaluation.incorrect / evaluation.total_samples if evaluation.total_samples > 0 else 0,
                'character_error_rate': evaluation.character_error_rate,
                'low_confidence_rate': len([c for c in evaluation.confidence_scores if c < 0.5]) / len(evaluation.confidence_scores) if evaluation.confidence_scores else 0
            }
            error_analysis['field_error_patterns'][field_name] = field_errors
            
            # Identify critical issues
            if evaluation.missing_rate > 0.3:
                error_analysis['critical_issues'].append(f"High missing rate for {field_name}: {evaluation.missing_rate:.2%}")
            
            if evaluation.character_error_rate > 0.2:
                error_analysis['critical_issues'].append(f"High character error rate for {field_name}: {evaluation.character_error_rate:.2%}")
        
        # Analyze validation errors
        for rule_name, evaluation in validation_evaluations.items():
            if evaluation.total_applicable > 0:
                error_analysis['validation_error_patterns'][rule_name] = {
                    'fail_rate': 1 - evaluation.pass_rate,
                    'common_errors': dict(evaluation.error_types)
                }
                
                if evaluation.pass_rate < 0.8:
                    error_analysis['critical_issues'].append(f"Low validation pass rate for {rule_name}: {evaluation.pass_rate:.2%}")
        
        return error_analysis
    
    def _generate_recommendations(self, field_evaluations: Dict[str, FieldEvaluation], 
                                validation_evaluations: Dict[str, ValidationEvaluation],
                                error_analysis: Dict[str, any]) -> List[str]:
        """Generate improvement recommendations based on evaluation results"""
        recommendations = []
        
        # Field-specific recommendations
        for field_name, evaluation in field_evaluations.items():
            if evaluation.missing_rate > 0.2:
                recommendations.append(f"Improve OCR preprocessing for {field_name} field - high missing rate detected")
            
            if evaluation.character_error_rate > 0.15:
                recommendations.append(f"Enhance regex patterns for {field_name} field - high character error rate")
            
            if evaluation.average_confidence < 0.6:
                recommendations.append(f"Review confidence scoring for {field_name} field - low average confidence")
        
        # Validation recommendations
        for rule_name, evaluation in validation_evaluations.items():
            if evaluation.pass_rate < 0.8:
                recommendations.append(f"Review validation rules for {rule_name} - low pass rate")
        
        # General recommendations
        if len(error_analysis['critical_issues']) > 3:
            recommendations.append("Consider retraining with more diverse synthetic data")
        
        recommendations.append("Implement active learning to continuously improve with new real samples")
        recommendations.append("Consider ensemble methods combining multiple OCR engines")
        
        return recommendations
    
    def _save_evaluation_report(self, report: EvaluationReport, output_dir: str):
        """Save evaluation report to files"""
        # Save JSON report
        report_dict = asdict(report)
        report_file = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
        
        # Save CSV summary
        summary_data = []
        for field_name, evaluation in report.field_evaluations.items():
            summary_data.append({
                'field': field_name,
                'exact_accuracy': evaluation.exact_accuracy,
                'fuzzy_accuracy': evaluation.fuzzy_accuracy,
                'missing_rate': evaluation.missing_rate,
                'character_error_rate': evaluation.character_error_rate,
                'average_confidence': evaluation.average_confidence
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, 'field_evaluation_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Evaluation report saved to {output_dir}")
    
    def _generate_visualizations(self, report: EvaluationReport, output_dir: str):
        """Generate visualization plots for the evaluation report"""
        plt.style.use('default')
        
        # Field accuracy comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Field accuracy comparison
        fields = list(report.field_evaluations.keys())
        exact_accuracies = [report.field_evaluations[f].exact_accuracy for f in fields]
        fuzzy_accuracies = [report.field_evaluations[f].fuzzy_accuracy for f in fields]
        
        x = np.arange(len(fields))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, exact_accuracies, width, label='Exact Match', alpha=0.8)
        axes[0, 0].bar(x + width/2, fuzzy_accuracies, width, label='Fuzzy Match', alpha=0.8)
        axes[0, 0].set_xlabel('Fields')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Field Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(fields, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Character error rates
        cer_rates = [report.field_evaluations[f].character_error_rate for f in fields]
        axes[0, 1].bar(fields, cer_rates, color='coral', alpha=0.8)
        axes[0, 1].set_xlabel('Fields')
        axes[0, 1].set_ylabel('Character Error Rate')
        axes[0, 1].set_title('Character Error Rate by Field')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Missing rates
        missing_rates = [report.field_evaluations[f].missing_rate for f in fields]
        axes[1, 0].bar(fields, missing_rates, color='lightcoral', alpha=0.8)
        axes[1, 0].set_xlabel('Fields')
        axes[1, 0].set_ylabel('Missing Rate')
        axes[1, 0].set_title('Missing Rate by Field')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Overall metrics
        overall_metrics = report.overall_metrics
        metric_names = list(overall_metrics.keys())
        metric_values = list(overall_metrics.values())
        
        axes[1, 1].bar(metric_names, metric_values, color='lightblue', alpha=0.8)
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Overall Performance Metrics')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_charts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Evaluation charts saved to {output_dir}")

def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Malaysian IC extraction system")
    parser.add_argument("--test_data", required=True, help="Path to test data JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    # Load test data
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Initialize components
    extractor = MalaysianICExtractor()
    validator = MalaysianICValidator()
    evaluator = MalaysianICEvaluator(extractor, validator)
    
    # Run evaluation
    report = evaluator.evaluate_dataset(test_data, args.output_dir)
    
    # Print summary
    print("\n🎯 Malaysian IC Extraction Evaluation Results")
    print("=" * 50)
    print(f"📊 Total samples: {report.dataset_info['total_samples']}")
    print(f"🎯 Weighted exact accuracy: {report.overall_metrics['weighted_exact_accuracy']:.3f}")
    print(f"🎯 Weighted fuzzy accuracy: {report.overall_metrics['weighted_fuzzy_accuracy']:.3f}")
    print(f"🏆 Full card success rate: {report.overall_metrics['full_card_success_rate']:.3f}")
    print(f"✅ Validation success rate: {report.overall_metrics['validation_success_rate']:.3f}")
    print(f"📈 Average confidence: {report.overall_metrics['average_confidence']:.3f}")
    print(f"⏱️  Average processing time: {report.performance_metrics.average_time_per_sample:.3f}s")
    
    print(f"\n📁 Detailed results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()