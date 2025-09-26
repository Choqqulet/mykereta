#!/usr/bin/env python3
"""
Complete Document Processing Pipeline for Malaysian Vehicle Registration Certificates

This module provides a comprehensive end-to-end pipeline that integrates:
1. Improved OCR preprocessing and text detection
2. Layout-aware field extraction with spatial analysis
3. Post-processing validation and formatting
4. Results visualization and export
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import re
from pathlib import Path
import argparse
from datetime import datetime
import os

# Import our custom modules
from improved_ocr_pipeline import ImprovedOCRPipeline, DocumentLayout
from layout_aware_field_extractor import TemplateBasedExtractor, ExtractedField

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Represents the result of field validation."""
    field_name: str
    original_value: str
    validated_value: str
    is_valid: bool
    validation_notes: List[str]
    confidence_adjustment: float = 0.0

@dataclass
class ProcessingResult:
    """Represents the complete processing result."""
    image_path: str
    processing_timestamp: str
    ocr_results: DocumentLayout
    extracted_fields: List[ExtractedField]
    validated_fields: Dict[str, ValidationResult]
    processing_stats: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class MalaysianDocumentValidator:
    """Validates extracted fields according to Malaysian document standards."""
    
    def __init__(self):
        # Malaysian-specific validation patterns
        self.validation_patterns = {
            'plate_number': {
                'pattern': r'^[A-Z]{1,3}\s*\d{1,4}\s*[A-Z]?$',
                'description': 'Malaysian vehicle plate format (e.g., WPR9256, ABC1234D)',
                'cleanup_rules': [
                    (r'\s+', ''),  # Remove spaces
                    (r'[^A-Z0-9]', ''),  # Remove non-alphanumeric
                ]
            },
            'nric': {
                'pattern': r'^\d{6}-\d{2}-\d{4}$',
                'description': 'Malaysian NRIC format (YYMMDD-PB-NNNN)',
                'cleanup_rules': [
                    (r'[^\d\-]', ''),  # Keep only digits and dashes
                ]
            },
            'year': {
                'pattern': r'^(19|20)\d{2}$',
                'description': 'Valid year (1900-2099)',
                'cleanup_rules': [
                    (r'[^\d]', ''),  # Keep only digits
                ]
            },
            'engine_capacity': {
                'pattern': r'^\d+(\.\d+)?\s*(cc|CC|l|L)?$',
                'description': 'Engine capacity in cc or liters',
                'cleanup_rules': [
                    (r'[^\d\.]', ''),  # Keep only digits and decimal point
                ]
            }
        }
        
        # Malaysian state codes for validation
        self.malaysian_states = {
            'A': 'Perak', 'B': 'Selangor', 'C': 'Pahang', 'D': 'Kelantan',
            'F': 'Putrajaya', 'G': 'Penang', 'H': 'Perlis', 'J': 'Johor',
            'K': 'Kedah', 'L': 'Labuan', 'M': 'Malacca', 'N': 'Negeri Sembilan',
            'P': 'Penang', 'Q': 'Sarawak', 'R': 'Perlis', 'S': 'Sabah',
            'T': 'Terengganu', 'V': 'Kuala Lumpur', 'W': 'Kuala Lumpur'
        }
    
    def validate_field(self, field: ExtractedField) -> ValidationResult:
        """Validate a single extracted field."""
        field_name = field.field_name
        original_value = field.value_region.text.strip()
        
        # Initialize validation result
        validation_result = ValidationResult(
            field_name=field_name,
            original_value=original_value,
            validated_value=original_value,
            is_valid=True,
            validation_notes=[]
        )
        
        # Apply field-specific validation
        if field_name in self.validation_patterns:
            validation_result = self._validate_with_pattern(validation_result, field_name)
        
        # Apply field-specific business logic
        if field_name == 'plate_number':
            validation_result = self._validate_plate_number(validation_result)
        elif field_name == 'year':
            validation_result = self._validate_year(validation_result)
        elif field_name == 'nric':
            validation_result = self._validate_nric(validation_result)
        
        return validation_result
    
    def _validate_with_pattern(self, result: ValidationResult, field_name: str) -> ValidationResult:
        """Validate field using regex pattern."""
        pattern_info = self.validation_patterns[field_name]
        pattern = pattern_info['pattern']
        cleanup_rules = pattern_info.get('cleanup_rules', [])
        
        # Apply cleanup rules
        cleaned_value = result.original_value
        for rule_pattern, replacement in cleanup_rules:
            cleaned_value = re.sub(rule_pattern, replacement, cleaned_value)
        
        result.validated_value = cleaned_value
        
        # Check pattern match
        if re.match(pattern, cleaned_value):
            result.validation_notes.append(f"Matches {pattern_info['description']}")
        else:
            result.is_valid = False
            result.validation_notes.append(f"Does not match expected format: {pattern_info['description']}")
            result.confidence_adjustment = -0.2
        
        return result
    
    def _validate_plate_number(self, result: ValidationResult) -> ValidationResult:
        """Validate Malaysian vehicle plate number."""
        plate = result.validated_value.upper()
        
        # Check if it starts with a valid state code
        if plate and plate[0] in self.malaysian_states:
            state_name = self.malaysian_states[plate[0]]
            result.validation_notes.append(f"Valid state code: {plate[0]} ({state_name})")
        else:
            result.validation_notes.append("Unknown or invalid state code")
            result.confidence_adjustment -= 0.1
        
        return result
    
    def _validate_year(self, result: ValidationResult) -> ValidationResult:
        """Validate vehicle year."""
        try:
            year = int(result.validated_value)
            current_year = datetime.now().year
            
            if 1900 <= year <= current_year:
                result.validation_notes.append(f"Valid vehicle year: {year}")
            elif year > current_year:
                result.is_valid = False
                result.validation_notes.append(f"Future year not allowed: {year}")
                result.confidence_adjustment = -0.3
            else:
                result.validation_notes.append(f"Very old vehicle year: {year}")
                result.confidence_adjustment -= 0.1
        except ValueError:
            result.is_valid = False
            result.validation_notes.append("Invalid year format")
            result.confidence_adjustment = -0.5
        
        return result
    
    def _validate_nric(self, result: ValidationResult) -> ValidationResult:
        """Validate Malaysian NRIC format and check digit."""
        nric = result.validated_value
        
        if len(nric) == 14 and nric[6] == '-' and nric[9] == '-':
            # Extract components
            birth_date = nric[:6]
            place_birth = nric[7:9]
            sequence = nric[10:14]
            
            # Validate birth date
            try:
                year = int(birth_date[:2])
                month = int(birth_date[2:4])
                day = int(birth_date[4:6])
                
                # Assume years 00-30 are 2000s, 31-99 are 1900s
                full_year = 2000 + year if year <= 30 else 1900 + year
                
                if 1 <= month <= 12 and 1 <= day <= 31:
                    result.validation_notes.append(f"Valid birth date: {day:02d}/{month:02d}/{full_year}")
                else:
                    result.validation_notes.append("Invalid birth date in NRIC")
                    result.confidence_adjustment -= 0.2
            except ValueError:
                result.validation_notes.append("Invalid birth date format in NRIC")
                result.confidence_adjustment -= 0.2
        
        return result

class CompleteDocumentPipeline:
    """Complete end-to-end document processing pipeline."""
    
    def __init__(self, template_path: Optional[str] = None):
        self.ocr_pipeline = ImprovedOCRPipeline(template_path)
        self.field_extractor = TemplateBasedExtractor(template_path)
        self.validator = MalaysianDocumentValidator()
        self.template_path = template_path
    
    def process_document(self, image_path: str, output_dir: Optional[str] = None) -> ProcessingResult:
        """Process a document image end-to-end."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting complete document processing for: {image_path}")
            
            # Step 1: OCR Processing
            logger.info("Step 1: Running improved OCR pipeline...")
            ocr_layout = self.ocr_pipeline.process_image(image_path, output_dir)
            
            # Convert layout to OCR results format for field extractor
            ocr_results = {
                'image_path': image_path,
                'image_size': ocr_layout.image_size,
                'preprocessing_method': ocr_layout.preprocessing_params.get('method', 'unknown'),
                'all_text_regions': [
                    {
                        'text': region.text,
                        'bbox': asdict(region.bbox),
                        'confidence': region.confidence,
                        'level': region.level
                    }
                    for region in ocr_layout.text_regions
                ]
            }
            
            # Step 2: Field Extraction
            logger.info("Step 2: Running layout-aware field extraction...")
            extracted_fields = self.field_extractor.extract_fields(ocr_results)
            
            # Step 3: Validation
            logger.info("Step 3: Validating extracted fields...")
            validated_fields = {}
            for field in extracted_fields:
                validation_result = self.validator.validate_field(field)
                validated_fields[field.field_name] = validation_result
                
                # Adjust field confidence based on validation
                if validation_result.confidence_adjustment != 0:
                    field.confidence = max(0, field.confidence + validation_result.confidence_adjustment * 100)
            
            # Step 4: Generate processing stats
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_stats = {
                'processing_time_seconds': processing_time,
                'total_text_regions': len(ocr_layout.text_regions),
                'extracted_fields_count': len(extracted_fields),
                'valid_fields_count': sum(1 for v in validated_fields.values() if v.is_valid),
                'preprocessing_method': ocr_layout.preprocessing_params.get('method', 'unknown'),
                'average_confidence': np.mean([f.confidence for f in extracted_fields]) if extracted_fields else 0
            }
            
            # Create processing result
            result = ProcessingResult(
                image_path=image_path,
                processing_timestamp=datetime.now().isoformat(),
                ocr_results=ocr_layout,
                extracted_fields=extracted_fields,
                validated_fields=validated_fields,
                processing_stats=processing_stats,
                success=True
            )
            
            # Step 5: Save results
            if output_dir:
                self._save_complete_results(result, output_dir)
            
            logger.info(f"Document processing completed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            return ProcessingResult(
                image_path=image_path,
                processing_timestamp=datetime.now().isoformat(),
                ocr_results=None,
                extracted_fields=[],
                validated_fields={},
                processing_stats={},
                success=False,
                error_message=str(e)
            )
    
    def _save_complete_results(self, result: ProcessingResult, output_dir: str):
        """Save complete processing results to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(result.image_path).stem
        
        # Save comprehensive results
        complete_results = {
            'document_info': {
                'image_path': result.image_path,
                'processing_timestamp': result.processing_timestamp,
                'success': result.success,
                'error_message': result.error_message
            },
            'processing_stats': result.processing_stats,
            'extracted_fields': {
                field.field_name: {
                    'text': field.value_region.text,
                    'confidence': field.confidence,
                    'extraction_method': field.extraction_method,
                    'bbox': asdict(field.value_region.bbox),
                    'spatial_context': field.spatial_context,
                    'label_text': field.label_region.text if field.label_region else None
                }
                for field in result.extracted_fields
            },
            'validation_results': {
                name: {
                    'original_value': validation.original_value,
                    'validated_value': validation.validated_value,
                    'is_valid': validation.is_valid,
                    'validation_notes': validation.validation_notes,
                    'confidence_adjustment': validation.confidence_adjustment
                }
                for name, validation in result.validated_fields.items()
            }
        }
        
        # Save JSON results
        with open(output_path / f"{image_name}_complete_results.json", 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
        
        # Create summary report
        self._create_summary_report(result, output_path / f"{image_name}_summary_report.txt")
        
        logger.info(f"Complete results saved to {output_dir}")
    
    def _create_summary_report(self, result: ProcessingResult, report_path: Path):
        """Create a human-readable summary report."""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MALAYSIAN VEHICLE REGISTRATION CERTIFICATE - PROCESSING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Document: {result.image_path}\n")
            f.write(f"Processed: {result.processing_timestamp}\n")
            f.write(f"Status: {'SUCCESS' if result.success else 'FAILED'}\n")
            if result.error_message:
                f.write(f"Error: {result.error_message}\n")
            f.write("\n")
            
            # Processing Statistics
            f.write("PROCESSING STATISTICS\n")
            f.write("-" * 30 + "\n")
            stats = result.processing_stats
            f.write(f"Processing Time: {stats.get('processing_time_seconds', 0):.2f} seconds\n")
            f.write(f"Text Regions Detected: {stats.get('total_text_regions', 0)}\n")
            f.write(f"Fields Extracted: {stats.get('extracted_fields_count', 0)}\n")
            f.write(f"Valid Fields: {stats.get('valid_fields_count', 0)}\n")
            f.write(f"Average Confidence: {stats.get('average_confidence', 0):.1f}%\n")
            f.write(f"Preprocessing Method: {stats.get('preprocessing_method', 'unknown')}\n\n")
            
            # Extracted Fields
            f.write("EXTRACTED FIELDS\n")
            f.write("-" * 30 + "\n")
            for field in result.extracted_fields:
                validation = result.validated_fields.get(field.field_name)
                status = "✓ VALID" if validation and validation.is_valid else "✗ INVALID"
                
                f.write(f"{field.field_name.upper().replace('_', ' ')}: {field.value_region.text}\n")
                f.write(f"  Status: {status}\n")
                f.write(f"  Confidence: {field.confidence:.1f}%\n")
                f.write(f"  Method: {field.extraction_method}\n")
                
                if validation:
                    if validation.validated_value != validation.original_value:
                        f.write(f"  Cleaned Value: {validation.validated_value}\n")
                    if validation.validation_notes:
                        f.write(f"  Notes: {'; '.join(validation.validation_notes)}\n")
                f.write("\n")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Complete Document Processing Pipeline for Malaysian Vehicle Registration Certificates")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--template", help="Path to document template JSON file")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = CompleteDocumentPipeline(template_path=args.template)
    
    # Process document
    result = pipeline.process_document(args.image_path, args.output)
    
    # Print results
    if result.success:
        print(f"\n✓ Document processing completed successfully!")
        print(f"Processing time: {result.processing_stats.get('processing_time_seconds', 0):.2f} seconds")
        print(f"Fields extracted: {len(result.extracted_fields)}")
        print(f"Valid fields: {sum(1 for v in result.validated_fields.values() if v.is_valid)}")
        
        print("\nExtracted and Validated Fields:")
        print("-" * 40)
        for field in result.extracted_fields:
            validation = result.validated_fields.get(field.field_name)
            status = "✓" if validation and validation.is_valid else "✗"
            print(f"{status} {field.field_name.replace('_', ' ').title()}: {field.value_region.text}")
        
        if args.output:
            print(f"\nDetailed results saved to: {args.output}")
    else:
        print(f"\n✗ Document processing failed: {result.error_message}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())