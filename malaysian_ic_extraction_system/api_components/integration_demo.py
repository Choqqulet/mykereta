#!/usr/bin/env python3
"""
Complete Integration Demo

This script demonstrates the full pipeline integration:
1. Document processing with improved OCR and layout analysis
2. Enhanced validation with Malaysian document rules
3. Quality assurance and review identification
4. Annotation export to multiple formats
5. Active learning pipeline setup
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, Any

# Import our pipeline components
from complete_document_pipeline import CompleteDocumentPipeline
from enhanced_validation_engine import EnhancedMalaysianValidator
from annotation_format_exporter import AnnotationFormatExporter
from quality_assurance_pipeline import QualityAssuranceEngine, ActiveLearningPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedDocumentProcessor:
    """Integrated document processor with all pipeline components."""
    
    def __init__(self, template_path: str):
        self.template_path = template_path
        
        # Initialize pipeline components
        self.document_pipeline = CompleteDocumentPipeline(template_path)
        self.enhanced_validator = EnhancedMalaysianValidator()
        self.annotation_exporter = AnnotationFormatExporter()
        self.qa_engine = QualityAssuranceEngine(confidence_threshold=0.75)
        self.active_learning = ActiveLearningPipeline(self.qa_engine)
        
        # Create output directories
        self.output_dir = Path("integrated_pipeline_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.annotations_dir = self.output_dir / "annotations"
        self.annotations_dir.mkdir(exist_ok=True)
        
        self.qa_dir = self.output_dir / "quality_assurance"
        self.qa_dir.mkdir(exist_ok=True)
    
    def process_document_complete(self, image_path: str, export_annotations: bool = True) -> Dict[str, Any]:
        """Process a document through the complete integrated pipeline."""
        logger.info(f"Starting integrated processing for: {image_path}")
        
        # Step 1: Run the complete document pipeline
        logger.info("Step 1: Running document processing pipeline...")
        pipeline_results = self.document_pipeline.process_document(
            image_path, 
            output_dir=str(self.output_dir / "pipeline_results")
        )
        
        # Step 2: Enhanced validation
        logger.info("Step 2: Running enhanced validation...")
        enhanced_validation_results = {}
        
        for field in pipeline_results.extracted_fields:
            validation_result = self.enhanced_validator.validate_field_enhanced(
                field.field_name,
                field.value_region.text,
                field.confidence
            )
            enhanced_validation_results[field.field_name] = {
                'is_valid': validation_result.is_valid,
                'confidence_score': {
                    'ocr_confidence': validation_result.confidence_score.ocr_confidence,
                    'model_confidence': validation_result.confidence_score.model_confidence,
                    'spatial_confidence': validation_result.confidence_score.spatial_confidence,
                    'regex_confidence': validation_result.confidence_score.regex_confidence,
                    'final_confidence': validation_result.confidence_score.final_confidence,
                    'confidence_factors': validation_result.confidence_score.confidence_factors
                },
                'validation_rules': validation_result.validation_rules_applied,
                'validation_notes': validation_result.validation_notes
            }
        
        # Convert ProcessingResult to dictionary format for compatibility
        pipeline_results_dict = {
            'document_info': {
                'image_path': pipeline_results.image_path,
                'processing_timestamp': pipeline_results.processing_timestamp,
                'success': pipeline_results.success
            },
            'processing_stats': pipeline_results.processing_stats,
            'extracted_fields': {
                 field.field_name: {
                     'text': field.value_region.text,
                     'confidence': field.confidence,
                     'bbox': {
                         'x': field.value_region.bbox.x,
                         'y': field.value_region.bbox.y,
                         'width': field.value_region.bbox.width,
                         'height': field.value_region.bbox.height
                     },
                     'extraction_method': field.extraction_method,
                     'spatial_context': field.spatial_context
                 } for field in pipeline_results.extracted_fields
             },
            'validation_results': {
                name: {
                    'is_valid': result.is_valid,
                    'validation_notes': result.validation_notes
                } for name, result in pipeline_results.validated_fields.items()
            },
            'enhanced_validation': enhanced_validation_results
        }
        
        # Step 3: Quality assurance
        logger.info("Step 3: Running quality assurance...")
        qa_results = self.qa_engine.process_document_results(pipeline_results_dict)
        
        # Step 4: Export annotations if requested
        annotation_paths = {}
        if export_annotations:
            logger.info("Step 4: Exporting annotations to multiple formats...")
            annotation_paths = self._export_annotations(pipeline_results_dict, image_path)
        
        # Step 5: Prepare comprehensive results
        integrated_results = {
            'document_info': pipeline_results_dict.get('document_info', {}),
            'processing_stats': pipeline_results_dict.get('processing_stats', {}),
            'extracted_fields': pipeline_results_dict.get('extracted_fields', {}),
            'validation_results': pipeline_results_dict.get('validation_results', {}),
            'enhanced_validation': enhanced_validation_results,
            'quality_assurance': qa_results,
            'annotation_exports': annotation_paths,
            'integration_metadata': {
                'pipeline_version': '2.0',
                'components_used': [
                    'complete_document_pipeline',
                    'enhanced_validation_engine',
                    'quality_assurance_pipeline',
                    'annotation_format_exporter'
                ],
                'processing_timestamp': datetime.now().isoformat()
            }
        }
        
        # Save integrated results
        results_file = self.output_dir / f"{Path(image_path).stem}_integrated_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(integrated_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Integrated processing complete. Results saved to: {results_file}")
        return integrated_results
    
    def _export_annotations(self, pipeline_results: Dict[str, Any], image_path: str) -> Dict[str, str]:
        """Export annotations to multiple formats."""
        # Convert pipeline results to annotation format
        annotation = self.annotation_exporter.convert_processing_result_to_annotation(pipeline_results)
        
        # Update with actual image dimensions (simplified)
        annotation.image_width = 800  # Would get from actual image
        annotation.image_height = 600  # Would get from actual image
        
        base_name = Path(image_path).stem
        annotation_paths = {}
        
        # Export to different formats
        formats = [
            ('labelimg_xml', lambda: self.annotation_exporter.export_to_labelimg(
                annotation, str(self.annotations_dir / f"{base_name}.xml"))),
            ('custom_json', lambda: self.annotation_exporter.export_to_custom_json(
                annotation, str(self.annotations_dir / f"{base_name}_custom.json"))),
            ('coco_json', lambda: self.annotation_exporter.export_to_coco(
                [annotation], str(self.annotations_dir / f"{base_name}_coco.json"))),
            ('csv', lambda: self.annotation_exporter.export_to_csv(
                [annotation], str(self.annotations_dir / f"{base_name}.csv")))
        ]
        
        for format_name, export_func in formats:
            try:
                path = export_func()
                annotation_paths[format_name] = path
                logger.info(f"Exported {format_name}: {path}")
            except Exception as e:
                logger.error(f"Failed to export {format_name}: {e}")
                annotation_paths[format_name] = f"Error: {e}"
        
        return annotation_paths
    
    def generate_quality_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive quality report."""
        qa_results = results.get('quality_assurance', {})
        quality_summary = qa_results.get('quality_summary', {})
        
        report_lines = [
            "INTEGRATED PIPELINE QUALITY REPORT",
            "=" * 50,
            "",
            f"Document: {results.get('document_info', {}).get('image_path', 'Unknown')}",
            f"Processing Time: {results.get('processing_stats', {}).get('processing_time_seconds', 0):.2f} seconds",
            f"Timestamp: {results.get('integration_metadata', {}).get('processing_timestamp', 'Unknown')}",
            "",
            "EXTRACTION SUMMARY:",
            f"  Total Fields Detected: {quality_summary.get('total_fields', 0)}",
            f"  High Quality Fields: {quality_summary.get('high_quality', 0)}",
            f"  Fields Needing Review: {quality_summary.get('needs_review', 0)}",
            f"  Critical Issues: {quality_summary.get('critical_issues', 0)}",
            f"  Overall Quality Score: {qa_results.get('overall_quality_score', 0):.2f}",
            "",
            "FIELD DETAILS:",
        ]
        
        # Add field details
        extracted_fields = results.get('extracted_fields', {})
        enhanced_validation = results.get('enhanced_validation', {})
        
        for field_name, field_data in extracted_fields.items():
            validation = enhanced_validation.get(field_name, {})
            
            report_lines.extend([
                f"  {field_name.upper()}:",
                f"    Text: '{field_data.get('text', '')}'",
                f"    Confidence: {field_data.get('confidence', 0):.1f}%",
                f"    Validation: {'✓ Valid' if validation.get('is_valid', False) else '✗ Invalid'}",
                f"    Enhanced Confidence: {validation.get('confidence_score', {}).get('final_confidence', 0):.2f}",
                f"    Rules Applied: {', '.join(validation.get('validation_rules', []))}",
                ""
            ])
        
        # Add review items
        review_items = qa_results.get('review_items', [])
        if review_items:
            report_lines.extend([
                "ITEMS REQUIRING REVIEW:",
                ""
            ])
            
            for item in review_items:
                report_lines.extend([
                    f"  {item['field_name'].upper()}:",
                    f"    Predicted: '{item['text']}'",
                    f"    Confidence: {item['confidence']:.1f}%",
                    f"    Quality Score: {item['quality_score']:.2f}",
                    f"    Issues: {', '.join(item['issues'])}",
                    ""
                ])
        
        # Add annotation export info
        annotation_exports = results.get('annotation_exports', {})
        if annotation_exports:
            report_lines.extend([
                "ANNOTATION EXPORTS:",
                ""
            ])
            
            for format_name, path in annotation_exports.items():
                status = "✓" if not path.startswith("Error:") else "✗"
                report_lines.append(f"  {status} {format_name}: {path}")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.qa_dir / f"{Path(results.get('document_info', {}).get('image_path', 'unknown')).stem}_quality_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Quality report saved to: {report_file}")
        return str(report_file)

def main():
    """Run the integrated pipeline demo."""
    # Configuration
    image_path = "sijil-pemilikan-kenderaan.jpeg"
    template_path = "templates/vehicle_registration_manual_template.json"
    
    # Check if files exist
    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        return
    
    if not Path(template_path).exists():
        logger.error(f"Template file not found: {template_path}")
        return
    
    print("INTEGRATED DOCUMENT PROCESSING PIPELINE DEMO")
    print("=" * 60)
    print()
    
    # Initialize integrated processor
    processor = IntegratedDocumentProcessor(template_path)
    
    # Process document
    print("🚀 Starting integrated document processing...")
    results = processor.process_document_complete(image_path, export_annotations=True)
    
    # Generate quality report
    print("📊 Generating quality report...")
    report_path = processor.generate_quality_report(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    
    qa_summary = results.get('quality_assurance', {}).get('quality_summary', {})
    print(f"📄 Document: {Path(image_path).name}")
    print(f"⏱️  Processing Time: {results.get('processing_stats', {}).get('processing_time_seconds', 0):.2f}s")
    print(f"🎯 Fields Extracted: {qa_summary.get('total_fields', 0)}")
    print(f"✅ High Quality: {qa_summary.get('high_quality', 0)}")
    print(f"⚠️  Needs Review: {qa_summary.get('needs_review', 0)}")
    print(f"📈 Quality Score: {results.get('quality_assurance', {}).get('overall_quality_score', 0):.2f}")
    
    print(f"\n📁 Output Directory: {processor.output_dir}")
    print(f"📋 Quality Report: {report_path}")
    
    # Show annotation exports
    annotation_exports = results.get('annotation_exports', {})
    if annotation_exports:
        print(f"\n📝 Annotation Exports:")
        for format_name, path in annotation_exports.items():
            status = "✅" if not path.startswith("Error:") else "❌"
            print(f"   {status} {format_name}: {Path(path).name}")
    
    print("\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    main()