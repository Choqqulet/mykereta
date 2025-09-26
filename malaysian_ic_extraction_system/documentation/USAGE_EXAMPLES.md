# Usage Examples and Code Snippets

This document provides practical examples for using the complete document processing pipeline in various scenarios.

## 🚀 Basic Usage Examples

### 1. Simple Document Processing

```python
#!/usr/bin/env python3
"""
Basic document processing example
"""
from integration_demo import IntegratedDocumentProcessor

def process_single_document():
    # Initialize processor
    processor = IntegratedDocumentProcessor()
    
    # Process document
    image_path = "path/to/your/document.jpg"
    results = processor.process_document_complete(
        image_path=image_path,
        export_annotations=True
    )
    
    # Print summary
    print(f"Processing completed in {results['processing_stats']['processing_time']:.2f}s")
    print(f"Fields extracted: {len(results['extracted_fields'])}")
    print(f"Quality score: {results['quality_assurance']['overall_quality']:.2f}")
    
    return results

if __name__ == "__main__":
    results = process_single_document()
```

### 2. Batch Document Processing

```python
#!/usr/bin/env python3
"""
Batch processing example for multiple documents
"""
import os
from pathlib import Path
from integration_demo import IntegratedDocumentProcessor
import json

def process_document_batch(input_directory, output_directory):
    """Process all images in a directory"""
    
    processor = IntegratedDocumentProcessor()
    
    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    
    batch_results = []
    
    for file_path in Path(input_directory).iterdir():
        if file_path.suffix.lower() in image_extensions:
            print(f"Processing: {file_path.name}")
            
            try:
                # Process document
                results = processor.process_document_complete(
                    image_path=str(file_path),
                    export_annotations=False  # Skip for batch processing speed
                )
                
                # Save individual results
                output_file = Path(output_directory) / f"{file_path.stem}_results.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                batch_results.append({
                    'filename': file_path.name,
                    'status': 'success',
                    'quality_score': results['quality_assurance']['overall_quality'],
                    'fields_extracted': len(results['extracted_fields']),
                    'processing_time': results['processing_stats']['processing_time']
                })
                
                print(f"  ✅ Success - Quality: {results['quality_assurance']['overall_quality']:.2f}")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                batch_results.append({
                    'filename': file_path.name,
                    'status': 'error',
                    'error': str(e)
                })
    
    # Save batch summary
    summary_file = Path(output_directory) / "batch_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_documents': len(batch_results),
            'successful': len([r for r in batch_results if r['status'] == 'success']),
            'failed': len([r for r in batch_results if r['status'] == 'error']),
            'average_quality': sum(r.get('quality_score', 0) for r in batch_results if r['status'] == 'success') / max(1, len([r for r in batch_results if r['status'] == 'success'])),
            'results': batch_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nBatch processing complete. Results saved to: {output_directory}")
    return batch_results

if __name__ == "__main__":
    input_dir = "input_documents/"
    output_dir = "batch_results/"
    
    results = process_document_batch(input_dir, output_dir)
```

### 3. Quality-Based Processing Workflow

```python
#!/usr/bin/env python3
"""
Quality-based processing with automatic approval/review routing
"""
from integration_demo import IntegratedDocumentProcessor
from quality_assurance_pipeline import QualityAssuranceEngine
import json

class QualityBasedProcessor:
    def __init__(self, quality_threshold=0.8):
        self.processor = IntegratedDocumentProcessor()
        self.qa_engine = QualityAssuranceEngine()
        self.quality_threshold = quality_threshold
        
        # Results storage
        self.auto_approved = []
        self.needs_review = []
        self.critical_issues = []
    
    def process_with_quality_routing(self, image_path):
        """Process document and route based on quality"""
        
        # Process document
        results = self.processor.process_document_complete(
            image_path=image_path,
            export_annotations=False
        )
        
        quality_score = results['quality_assurance']['overall_quality']
        
        # Route based on quality
        if quality_score >= self.quality_threshold:
            self._auto_approve(results, image_path)
        elif quality_score >= 0.5:
            self._queue_for_review(results, image_path)
        else:
            self._flag_critical_issues(results, image_path)
        
        return results
    
    def _auto_approve(self, results, image_path):
        """Auto-approve high quality results"""
        self.auto_approved.append({
            'image_path': image_path,
            'quality_score': results['quality_assurance']['overall_quality'],
            'extracted_fields': results['extracted_fields'],
            'timestamp': results['document_info']['processing_timestamp']
        })
        print(f"✅ AUTO-APPROVED: {image_path} (Quality: {results['quality_assurance']['overall_quality']:.2f})")
    
    def _queue_for_review(self, results, image_path):
        """Queue for human review"""
        review_items = results['quality_assurance'].get('review_queue', [])
        
        self.needs_review.append({
            'image_path': image_path,
            'quality_score': results['quality_assurance']['overall_quality'],
            'review_items': review_items,
            'extracted_fields': results['extracted_fields'],
            'timestamp': results['document_info']['processing_timestamp']
        })
        print(f"⚠️  NEEDS REVIEW: {image_path} (Quality: {results['quality_assurance']['overall_quality']:.2f})")
    
    def _flag_critical_issues(self, results, image_path):
        """Flag critical quality issues"""
        self.critical_issues.append({
            'image_path': image_path,
            'quality_score': results['quality_assurance']['overall_quality'],
            'issues': results['quality_assurance'].get('critical_issues', []),
            'timestamp': results['document_info']['processing_timestamp']
        })
        print(f"🚨 CRITICAL ISSUES: {image_path} (Quality: {results['quality_assurance']['overall_quality']:.2f})")
    
    def generate_processing_report(self, output_path="processing_report.json"):
        """Generate comprehensive processing report"""
        report = {
            'summary': {
                'total_processed': len(self.auto_approved) + len(self.needs_review) + len(self.critical_issues),
                'auto_approved': len(self.auto_approved),
                'needs_review': len(self.needs_review),
                'critical_issues': len(self.critical_issues),
                'approval_rate': len(self.auto_approved) / max(1, len(self.auto_approved) + len(self.needs_review) + len(self.critical_issues))
            },
            'auto_approved': self.auto_approved,
            'needs_review': self.needs_review,
            'critical_issues': self.critical_issues
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Processing Report:")
        print(f"   Total Processed: {report['summary']['total_processed']}")
        print(f"   Auto-Approved: {report['summary']['auto_approved']}")
        print(f"   Needs Review: {report['summary']['needs_review']}")
        print(f"   Critical Issues: {report['summary']['critical_issues']}")
        print(f"   Approval Rate: {report['summary']['approval_rate']:.1%}")
        print(f"   Report saved to: {output_path}")
        
        return report

# Example usage
if __name__ == "__main__":
    processor = QualityBasedProcessor(quality_threshold=0.8)
    
    # Process multiple documents
    documents = [
        "document1.jpg",
        "document2.jpg", 
        "document3.jpg"
    ]
    
    for doc in documents:
        processor.process_with_quality_routing(doc)
    
    # Generate report
    processor.generate_processing_report()
```

### 4. Training Data Generation

```python
#!/usr/bin/env python3
"""
Generate training data with annotations for model improvement
"""
from integration_demo import IntegratedDocumentProcessor
from annotation_format_exporter import AnnotationFormatExporter
import json
from pathlib import Path

class TrainingDataGenerator:
    def __init__(self, output_directory="training_data"):
        self.processor = IntegratedDocumentProcessor()
        self.exporter = AnnotationFormatExporter()
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
    
    def generate_training_sample(self, image_path, copy_image=True):
        """Generate training annotations for a single image"""
        
        # Process document
        results = self.processor.process_document_complete(
            image_path=image_path,
            export_annotations=True
        )
        
        image_name = Path(image_path).stem
        
        # Copy image to training directory
        if copy_image:
            import shutil
            target_image = self.output_dir / "images" / Path(image_path).name
            shutil.copy2(image_path, target_image)
        
        # Export annotations in multiple formats
        annotation_data = self._prepare_annotation_data(results, image_path)
        
        # YOLO format for object detection
        yolo_path = self.output_dir / "labels" / f"{image_name}.txt"
        self._export_yolo_format(annotation_data, yolo_path)
        
        # COCO format for complex annotations
        coco_path = self.output_dir / "annotations" / f"{image_name}_coco.json"
        self._export_coco_format(annotation_data, coco_path)
        
        # Custom format for field extraction
        custom_path = self.output_dir / "annotations" / f"{image_name}_fields.json"
        self._export_field_format(results, custom_path)
        
        return {
            'image_path': image_path,
            'annotations': {
                'yolo': str(yolo_path),
                'coco': str(coco_path),
                'fields': str(custom_path)
            },
            'quality_score': results['quality_assurance']['overall_quality']
        }
    
    def _prepare_annotation_data(self, results, image_path):
        """Prepare annotation data from processing results"""
        annotations = []
        
        for field_name, field_data in results['extracted_fields'].items():
            bbox = field_data['bbox']
            annotations.append({
                'field_name': field_name,
                'text': field_data['text'],
                'bbox': bbox,
                'confidence': field_data['confidence']
            })
        
        return {
            'image_path': image_path,
            'annotations': annotations
        }
    
    def _export_yolo_format(self, annotation_data, output_path):
        """Export in YOLO format for object detection"""
        # This is a simplified example - you'd need image dimensions
        # and proper class mapping for real YOLO training
        
        with open(output_path, 'w') as f:
            for ann in annotation_data['annotations']:
                # YOLO format: class_id center_x center_y width height (normalized)
                # This is a placeholder - implement proper normalization
                f.write(f"0 0.5 0.5 0.1 0.05\\n")  # Placeholder values
    
    def _export_coco_format(self, annotation_data, output_path):
        """Export in COCO format"""
        coco_data = {
            'images': [{
                'id': 1,
                'file_name': Path(annotation_data['image_path']).name,
                'width': 1000,  # Placeholder - get actual dimensions
                'height': 1000
            }],
            'annotations': [],
            'categories': [
                {'id': i, 'name': field} 
                for i, field in enumerate(['plate_number', 'year', 'fuel_type', 'engine_capacity'], 1)
            ]
        }
        
        for i, ann in enumerate(annotation_data['annotations']):
            bbox = ann['bbox']
            coco_data['annotations'].append({
                'id': i + 1,
                'image_id': 1,
                'category_id': 1,  # Simplified - map field names to category IDs
                'bbox': [bbox['x'], bbox['y'], bbox['width'], bbox['height']],
                'area': bbox['width'] * bbox['height'],
                'iscrowd': 0
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    def _export_field_format(self, results, output_path):
        """Export field extraction training data"""
        training_data = {
            'image_info': results['document_info'],
            'fields': results['extracted_fields'],
            'validation': results['validation_results'],
            'enhanced_validation': results['enhanced_validation']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    def generate_dataset_config(self):
        """Generate dataset configuration files"""
        
        # YOLO dataset config
        yolo_config = {
            'train': str(self.output_dir / "images"),
            'val': str(self.output_dir / "images"),
            'nc': 4,  # Number of classes
            'names': ['plate_number', 'year', 'fuel_type', 'engine_capacity']
        }
        
        with open(self.output_dir / "dataset.yaml", 'w') as f:
            import yaml
            yaml.dump(yolo_config, f)
        
        # Training metadata
        metadata = {
            'dataset_info': {
                'name': 'Malaysian Vehicle Registration Certificate Dataset',
                'description': 'Generated training data for document field extraction',
                'version': '1.0',
                'created': str(Path().cwd()),
                'format': 'Multiple (YOLO, COCO, Custom)'
            },
            'statistics': {
                'total_images': len(list((self.output_dir / "images").glob("*"))),
                'annotation_formats': ['YOLO', 'COCO', 'Custom Fields']
            }
        }
        
        with open(self.output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    generator = TrainingDataGenerator("training_dataset")
    
    # Generate training data for multiple images
    training_images = [
        "sample1.jpg",
        "sample2.jpg",
        "sample3.jpg"
    ]
    
    results = []
    for image_path in training_images:
        result = generator.generate_training_sample(image_path)
        results.append(result)
        print(f"Generated training data for: {image_path}")
    
    # Generate dataset configuration
    generator.generate_dataset_config()
    
    print(f"\\nTraining dataset generated in: {generator.output_dir}")
    print(f"Total samples: {len(results)}")
```

### 5. Custom Validation Rules

```python
#!/usr/bin/env python3
"""
Example of adding custom validation rules
"""
from enhanced_validation_engine import EnhancedMalaysianValidator, ValidationRule
from integration_demo import IntegratedDocumentProcessor

class CustomDocumentProcessor:
    def __init__(self):
        self.processor = IntegratedDocumentProcessor()
        self._setup_custom_validation()
    
    def _setup_custom_validation(self):
        """Add custom validation rules"""
        
        # Custom rule for engine number format
        engine_number_rule = ValidationRule(
            name='Engine Number Format',
            pattern=r'^[A-Z0-9]{8,17}$',
            description='Engine number should be 8-17 alphanumeric characters',
            confidence_boost=0.15,
            cleanup_rules=[
                (r'[^A-Z0-9]', ''),  # Remove non-alphanumeric
                (r'O', '0'),         # Common OCR error
                (r'I', '1'),         # Common OCR error
            ]
        )
        
        # Custom rule for chassis number
        chassis_number_rule = ValidationRule(
            name='Chassis Number Format',
            pattern=r'^[A-HJ-NPR-Z0-9]{17}$',
            description='17-character chassis number (VIN format)',
            confidence_boost=0.2,
            cleanup_rules=[
                (r'[^A-HJ-NPR-Z0-9]', ''),  # Remove invalid characters
                (r'O', '0'),                 # OCR corrections
                (r'I', '1'),
                (r'Q', '0'),
            ]
        )
        
        # Custom rule for registration date
        registration_date_rule = ValidationRule(
            name='Registration Date Format',
            pattern=r'^\\d{2}/\\d{2}/\\d{4}$',
            description='Date in DD/MM/YYYY format',
            confidence_boost=0.1,
            cleanup_rules=[
                (r'[^0-9/]', ''),  # Keep only digits and slashes
                (r'O', '0'),       # OCR corrections
            ]
        )
        
        # Add rules to validator
        validator = self.processor.enhanced_validator
        validator.validation_rules.update({
            'engine_number': engine_number_rule,
            'chassis_number': chassis_number_rule,
            'registration_date': registration_date_rule
        })
    
    def process_with_custom_validation(self, image_path):
        """Process document with custom validation rules"""
        
        results = self.processor.process_document_complete(
            image_path=image_path,
            export_annotations=True
        )
        
        # Additional custom validation logic
        self._apply_business_rules(results)
        
        return results
    
    def _apply_business_rules(self, results):
        """Apply custom business logic validation"""
        
        extracted_fields = results['extracted_fields']
        
        # Example: Cross-field validation
        if 'year' in extracted_fields and 'registration_date' in extracted_fields:
            year = extracted_fields['year']['text']
            reg_date = extracted_fields['registration_date']['text']
            
            try:
                # Extract year from registration date
                reg_year = reg_date.split('/')[-1]
                
                # Validate year consistency
                if abs(int(year) - int(reg_year)) > 1:
                    results['validation_warnings'] = results.get('validation_warnings', [])
                    results['validation_warnings'].append({
                        'type': 'cross_field_inconsistency',
                        'message': f'Vehicle year ({year}) and registration year ({reg_year}) are inconsistent',
                        'severity': 'warning'
                    })
            except (ValueError, IndexError):
                pass  # Skip if date parsing fails
        
        # Example: Range validation
        if 'engine_capacity' in extracted_fields:
            try:
                capacity = float(extracted_fields['engine_capacity']['text'])
                if capacity < 0.1 or capacity > 10.0:  # Reasonable range for cars
                    results['validation_warnings'] = results.get('validation_warnings', [])
                    results['validation_warnings'].append({
                        'type': 'value_out_of_range',
                        'message': f'Engine capacity ({capacity}L) seems unusual',
                        'severity': 'warning'
                    })
            except ValueError:
                pass  # Skip if not a valid number

# Example usage
if __name__ == "__main__":
    processor = CustomDocumentProcessor()
    
    results = processor.process_with_custom_validation("test_document.jpg")
    
    # Print validation results
    print("Validation Results:")
    for field_name, validation in results.get('enhanced_validation', {}).items():
        print(f"  {field_name}: {'✓' if validation['is_valid'] else '✗'}")
        if validation['validation_rules']:
            print(f"    Rules: {', '.join(validation['validation_rules'])}")
    
    # Print warnings if any
    if 'validation_warnings' in results:
        print("\\nValidation Warnings:")
        for warning in results['validation_warnings']:
            print(f"  ⚠️  {warning['message']}")
```

## 🔧 Configuration Examples

### Environment Configuration

```python
# config.py
import os
from pathlib import Path

class PipelineConfig:
    # Paths
    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = BASE_DIR / "pipeline_output"
    TEMP_DIR = BASE_DIR / "temp"
    
    # OCR Settings
    OCR_METHODS = ['conservative', 'balanced', 'aggressive']
    DEFAULT_OCR_METHOD = 'balanced'
    TESSERACT_CONFIG = '--oem 3 --psm 6'
    
    # Quality Thresholds
    HIGH_QUALITY_THRESHOLD = 0.85
    REVIEW_THRESHOLD = 0.70
    CRITICAL_THRESHOLD = 0.50
    
    # Validation Settings
    ENABLE_ENHANCED_VALIDATION = True
    ENABLE_CROSS_FIELD_VALIDATION = True
    
    # Export Settings
    EXPORT_FORMATS = ['coco', 'custom_json', 'csv']
    ENABLE_VISUALIZATION = True
    
    # Database Settings (for QA pipeline)
    QA_DATABASE_PATH = BASE_DIR / "qa_database.db"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
```

### Logging Configuration

```python
# logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set specific logger levels
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Usage
if __name__ == "__main__":
    setup_logging(
        log_level=logging.INFO,
        log_file="pipeline.log"
    )
```

These examples provide a comprehensive foundation for using the document processing pipeline in various real-world scenarios. Each example can be adapted and extended based on specific requirements and use cases.