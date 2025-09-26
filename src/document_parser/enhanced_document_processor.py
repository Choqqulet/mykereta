#!/usr/bin/env python3
"""
Enhanced Document Processor

Main document processing pipeline that integrates all enhanced components
for robust IC and Passport document processing.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

# Import our enhanced components
from .enhanced_ocr_service import EnhancedOCRService
from .intelligent_field_extractor import IntelligentFieldExtractor, ExtractionResult
from .validation_service import DocumentValidationService, ValidationResult
from .error_handler import DocumentErrorHandler, DocumentError, ProcessingResult, ErrorSeverity
from .malaysian_ic_ocr_integration import MalaysianICOCREngine, MalaysianICOCRResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentProcessingResult:
    """Complete document processing result"""
    success: bool
    document_type: str = "unknown"
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    quality_score: float = 0.0
    errors: List[DocumentError] = field(default_factory=list)
    warnings: List[DocumentError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_ocr_text: str = ""
    processing_stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class EnhancedDocumentProcessor:
    """
    Enhanced document processor for IC and Passport documents.
    
    Features:
    - Multi-format support (PDF, JPG, PNG)
    - Quality assessment and preprocessing
    - Multi-engine OCR with consensus
    - Intelligent field extraction
    - Comprehensive validation
    - Robust error handling and recovery
    - Detailed processing metrics
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 enable_quality_assessment: bool = True,
                 enable_auto_recovery: bool = True,
                 confidence_threshold: float = 0.6):
        """
        Initialize the enhanced document processor.
        
        Args:
            config: Configuration dictionary
            enable_quality_assessment: Enable document quality assessment
            enable_auto_recovery: Enable automatic error recovery
            confidence_threshold: Minimum confidence threshold for results
        """
        self.config = config or {}
        self.enable_quality_assessment = enable_quality_assessment
        self.enable_auto_recovery = enable_auto_recovery
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        try:
            # Initialize OCR service
            ocr_engines = self.config.get('ocr_engines', ['tesseract', 'easyocr'])
            self.ocr_service = EnhancedOCRService(
                engines=ocr_engines,
                languages=['en', 'ms'],
                enable_gpu=self.config.get('enable_gpu', False)
            )
            
            self.field_extractor = IntelligentFieldExtractor(
                confidence_threshold=confidence_threshold,
                enable_ml_extraction=self.config.get('enable_ml_extraction', True)
            )
            
            # Initialize validation service
            from src.document_parser.validation_service import ValidationLevel
            validation_level = ValidationLevel.STRICT if self.config.get('strict_validation', False) else ValidationLevel.MODERATE
            self.validator = DocumentValidationService(
                validation_level=validation_level
            )
            
            self.error_handler = DocumentErrorHandler(
                max_retries=self.config.get('max_retries', 3),
                enable_auto_recovery=enable_auto_recovery
            )
            
            # Initialize Malaysian IC OCR engine for specialized IC processing
            self.malaysian_ic_ocr = MalaysianICOCREngine()
            
            logger.info("Enhanced Document Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize document processor: {e}")
            raise
    
    def process_document(self, 
                        file_path: str,
                        document_type: str = "auto",
                        processing_options: Optional[Dict[str, Any]] = None) -> DocumentProcessingResult:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            document_type: Document type hint ('ic', 'passport', 'auto')
            processing_options: Additional processing options
            
        Returns:
            DocumentProcessingResult: Complete processing results
        """
        start_time = time.time()
        processing_stages = {}
        errors = []
        warnings = []
        
        try:
            logger.info(f"Starting document processing: {file_path}")
            
            # Stage 1: Input validation
            stage_start = time.time()
            is_valid, validation_errors = self.error_handler.validate_input_document(file_path)
            processing_stages['input_validation'] = {
                'duration': time.time() - stage_start,
                'success': is_valid,
                'errors': len(validation_errors)
            }
            
            if not is_valid:
                critical_errors = [e for e in validation_errors if e.severity == ErrorSeverity.CRITICAL]
                if critical_errors:
                    return DocumentProcessingResult(
                        success=False,
                        errors=validation_errors,
                        processing_time=time.time() - start_time,
                        processing_stages=processing_stages
                    )
            
            errors.extend([e for e in validation_errors if e.severity == ErrorSeverity.CRITICAL])
            warnings.extend([e for e in validation_errors if e.severity != ErrorSeverity.CRITICAL])
            
            # Stage 2: OCR processing
            stage_start = time.time()
            ocr_result = self._process_ocr(file_path, processing_options)
            processing_stages['ocr'] = {
                'duration': time.time() - stage_start,
                'success': ocr_result.get('success', False),
                'confidence': ocr_result.get('confidence', 0.0),
                'engines_used': ocr_result.get('engines_used', [])
            }
            
            if not ocr_result.get('success', False):
                error = DocumentError(
                    error_id="OCR_FAILED",
                    category=self.error_handler._classify_error(Exception("OCR processing failed"), {})[0],
                    severity=ErrorSeverity.HIGH,
                    message="OCR processing failed",
                    details=ocr_result.get('error', 'Unknown OCR error'),
                    file_path=file_path
                )
                errors.append(error)
                
                # Attempt recovery if enabled
                if self.enable_auto_recovery:
                    recovery_result = self.error_handler.attempt_recovery(error, {'file_path': file_path})
                    if not recovery_result.success:
                        return DocumentProcessingResult(
                            success=False,
                            errors=errors + [error],
                            warnings=warnings,
                            processing_time=time.time() - start_time,
                            processing_stages=processing_stages
                        )
            
            # Stage 3: Quality assessment
            quality_score = 1.0
            if self.enable_quality_assessment and 'image' in ocr_result:
                stage_start = time.time()
                quality_score, quality_issues = self.error_handler.assess_document_quality(
                    ocr_result['image'], file_path
                )
                processing_stages['quality_assessment'] = {
                    'duration': time.time() - stage_start,
                    'quality_score': quality_score,
                    'issues_found': len(quality_issues)
                }
                warnings.extend(quality_issues)
            
            # Stage 4: Field extraction
            stage_start = time.time()
            extraction_result = self.field_extractor.extract_fields(
                ocr_result,
                document_type=document_type,
                use_coordinates=processing_options.get('use_coordinates', True) if processing_options else True
            )
            processing_stages['field_extraction'] = {
                'duration': time.time() - stage_start,
                'success': len(extraction_result.extracted_fields) > 0,
                'fields_extracted': len(extraction_result.extracted_fields),
                'confidence': extraction_result.confidence_score,
                'methods_used': extraction_result.extraction_methods_used
            }
            
            # Stage 5: Validation
            stage_start = time.time()
            validation_result = self.validator.validate_document(
                extraction_result.structured_data,
                document_type=extraction_result.document_type
            )
            processing_stages['validation'] = {
                'duration': time.time() - stage_start,
                'overall_valid': validation_result.overall_valid,
                'validation_score': validation_result.confidence_score,
                'fields_validated': len(validation_result.field_results)
            }
            
            # Collect validation errors and warnings
            for field_name, field_validation in validation_result.field_results.items():
                if not field_validation.is_valid:
                    if field_validation.errors:
                        errors.append(DocumentError(
                            error_id=f"VALIDATION_{field_name.upper()}",
                            category=self.error_handler._classify_error(Exception("Validation failed"), {})[0],
                            severity=ErrorSeverity.HIGH,
                            message=f"Validation failed for {field_name}",
                            details=field_validation.error_message,
                            file_path=file_path
                        ))
                    else:
                        warnings.append(DocumentError(
                            error_id=f"VALIDATION_WARNING_{field_name.upper()}",
                            category=self.error_handler._classify_error(Exception("Validation warning"), {})[0],
                            severity=ErrorSeverity.MEDIUM,
                            message=f"Validation warning for {field_name}",
                            details=field_validation.error_message,
                            file_path=file_path
                        ))
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                extraction_result.confidence_score,
                validation_result.confidence_score,
                quality_score
            )
            
            # Determine success
            success = (
                overall_confidence >= self.confidence_threshold and
                len([e for e in errors if e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]]) == 0
            )
            
            # Create metadata
            metadata = {
                'processing_timestamp': datetime.now().isoformat(),
                'file_path': file_path,
                'file_size': Path(file_path).stat().st_size if Path(file_path).exists() else 0,
                'processor_version': '1.0.0',
                'processing_options': processing_options or {},
                'ocr_engines_used': ocr_result.get('engines_used', []),
                'extraction_methods_used': extraction_result.extraction_methods_used
            }
            
            return DocumentProcessingResult(
                success=success,
                document_type=extraction_result.document_type,
                extracted_fields=self._format_extracted_fields(extraction_result.extracted_fields),
                validation_results=self._format_validation_results(validation_result),
                confidence_score=overall_confidence,
                processing_time=time.time() - start_time,
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
                raw_ocr_text=extraction_result.raw_text,
                processing_stages=processing_stages
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            error = self.error_handler.handle_processing_error(e, {'operation': 'document_processing'}, file_path)
            
            return DocumentProcessingResult(
                success=False,
                errors=[error],
                processing_time=time.time() - start_time,
                processing_stages=processing_stages
            )
    
    def process_batch(self, 
                     file_paths: List[str],
                     document_type: str = "auto",
                     processing_options: Optional[Dict[str, Any]] = None,
                     max_workers: int = 4) -> Dict[str, DocumentProcessingResult]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            document_type: Document type hint
            processing_options: Processing options
            max_workers: Maximum number of worker threads
            
        Returns:
            Dict mapping file paths to processing results
        """
        results = {}
        
        try:
            logger.info(f"Starting batch processing of {len(file_paths)} documents")
            
            # For now, process sequentially (can be enhanced with threading)
            for file_path in file_paths:
                try:
                    result = self.process_document(file_path, document_type, processing_options)
                    results[file_path] = result
                    
                    logger.info(f"Processed {file_path}: {'SUCCESS' if result.success else 'FAILED'}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    error = self.error_handler.handle_processing_error(e, {'operation': 'batch_processing'}, file_path)
                    results[file_path] = DocumentProcessingResult(
                        success=False,
                        errors=[error]
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {}
    
    def get_processing_statistics(self, results: List[DocumentProcessingResult]) -> Dict[str, Any]:
        """
        Generate processing statistics from results.
        
        Args:
            results: List of processing results
            
        Returns:
            Dict containing processing statistics
        """
        try:
            if not results:
                return {}
            
            total_docs = len(results)
            successful_docs = len([r for r in results if r.success])
            
            # Calculate averages
            avg_confidence = sum(r.confidence_score for r in results) / total_docs
            avg_processing_time = sum(r.processing_time for r in results) / total_docs
            avg_quality_score = sum(r.quality_score for r in results) / total_docs
            
            # Document type distribution
            doc_types = {}
            for result in results:
                doc_type = result.document_type
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            # Error analysis
            all_errors = []
            for result in results:
                all_errors.extend(result.errors)
            
            error_report = self.error_handler.generate_error_report(all_errors)
            
            return {
                'summary': {
                    'total_documents': total_docs,
                    'successful_documents': successful_docs,
                    'success_rate': successful_docs / total_docs if total_docs > 0 else 0,
                    'average_confidence': avg_confidence,
                    'average_processing_time': avg_processing_time,
                    'average_quality_score': avg_quality_score
                },
                'document_types': doc_types,
                'error_analysis': error_report,
                'processing_stages': self._analyze_processing_stages(results)
            }
            
        except Exception as e:
            logger.error(f"Statistics generation failed: {e}")
            return {'error': 'Failed to generate statistics'}
    
    # Helper methods
    def _process_ocr(self, file_path: str, processing_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process OCR for the document"""
        try:
            # Determine preprocessing options
            preprocess_options = {}
            if processing_options:
                preprocess_options = processing_options.get('preprocessing', {})
            
            # Check if this is a Malaysian IC document and use specialized processing
            document_type = processing_options.get('document_type', 'auto') if processing_options else 'auto'
            use_malaysian_ic = (
                document_type == 'malaysian_ic' or 
                processing_options.get('use_malaysian_ic_ocr', False) if processing_options else False
            )
            
            if use_malaysian_ic:
                logger.info("Using Malaysian IC OCR engine for specialized processing")
                # Use Malaysian IC OCR engine
                malaysian_result = self.malaysian_ic_ocr.process_malaysian_ic(file_path)
                
                # Convert Malaysian IC result to standard format
                result = {
                    'success': malaysian_result.success,
                    'text': malaysian_result.combined_text,
                    'confidence': malaysian_result.confidence,
                    'quality_score': malaysian_result.quality_score,
                    'extracted_fields': malaysian_result.extracted_fields,
                    'tesseract_result': malaysian_result.tesseract_result,
                    'easyocr_result': malaysian_result.easyocr_result,
                    'preprocessing_applied': malaysian_result.preprocessing_applied,
                    'processing_time': malaysian_result.processing_time,
                    'metadata': {
                        'engine': 'malaysian_ic_ocr',
                        'preprocessing_steps': malaysian_result.preprocessing_applied
                    }
                }
            else:
                # Use standard OCR service
                result = self.ocr_service.process_document(
                    document_input=file_path,
                    document_type="auto",
                    quality_enhancement=True,
                    extract_coordinates=True
                )
            
            return result
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0.0
            }
    
    def _calculate_overall_confidence(self, 
                                    extraction_confidence: float,
                                    confidence_score: float,
                                    quality_score: float) -> float:
        """Calculate overall processing confidence"""
        try:
            # Weighted average of different confidence scores
            weights = {
                'extraction': 0.4,
                'validation': 0.4,
                'quality': 0.2
            }
            
            overall = (
                extraction_confidence * weights['extraction'] +
                confidence_score * weights['validation'] +
                quality_score * weights['quality']
            )
            
            return min(1.0, max(0.0, overall))
            
        except Exception:
            return 0.0
    
    def _format_extracted_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Format extracted fields for output"""
        formatted = {}
        
        try:
            for field_name, field_match in fields.items():
                formatted[field_name] = {
                    'value': field_match.value,
                    'confidence': field_match.confidence,
                    'extraction_method': field_match.method,
                    'alternatives': field_match.alternatives,
                    'bbox': field_match.bbox
                }
            
            return formatted
            
        except Exception as e:
            logger.error(f"Field formatting failed: {e}")
            return {}
    
    def _format_validation_results(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Format validation results for output"""
        try:
            formatted = {
                'overall_valid': validation_result.overall_valid,
                'validation_score': validation_result.confidence_score,
                'field_validations': {}
            }
            
            for field_name, field_validation in validation_result.field_results.items():
                formatted['field_validations'][field_name] = {
                    'is_valid': field_validation.is_valid,
                    'confidence': field_validation.confidence,
                    'errors': field_validation.errors,
                    'warnings': field_validation.warnings,
                    'suggestions': field_validation.suggestions
                }
            
            if validation_result.cross_field_errors:
                formatted['cross_validation'] = validation_result.cross_field_errors
            
            return formatted
            
        except Exception as e:
            logger.error(f"Validation result formatting failed: {e}")
            return {}
    
    def _analyze_processing_stages(self, results: List[DocumentProcessingResult]) -> Dict[str, Any]:
        """Analyze processing stages across multiple results"""
        try:
            stage_analysis = {}
            
            for result in results:
                for stage_name, stage_data in result.processing_stages.items():
                    if stage_name not in stage_analysis:
                        stage_analysis[stage_name] = {
                            'total_duration': 0.0,
                            'success_count': 0,
                            'total_count': 0,
                            'avg_duration': 0.0,
                            'success_rate': 0.0
                        }
                    
                    stage_info = stage_analysis[stage_name]
                    stage_info['total_duration'] += stage_data.get('duration', 0.0)
                    stage_info['total_count'] += 1
                    
                    if stage_data.get('success', False):
                        stage_info['success_count'] += 1
            
            # Calculate averages and rates
            for stage_name, stage_info in stage_analysis.items():
                if stage_info['total_count'] > 0:
                    stage_info['avg_duration'] = stage_info['total_duration'] / stage_info['total_count']
                    stage_info['success_rate'] = stage_info['success_count'] / stage_info['total_count']
            
            return stage_analysis
            
        except Exception as e:
            logger.error(f"Stage analysis failed: {e}")
            return {}
    
    def export_results(self, 
                      results: Union[DocumentProcessingResult, List[DocumentProcessingResult]],
                      output_path: str,
                      format: str = "json") -> bool:
        """
        Export processing results to file.
        
        Args:
            results: Processing result(s) to export
            output_path: Output file path
            format: Export format ('json', 'csv')
            
        Returns:
            bool: Success status
        """
        try:
            if isinstance(results, DocumentProcessingResult):
                results = [results]
            
            if format.lower() == "json":
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_documents': len(results),
                    'statistics': self.get_processing_statistics(results),
                    'results': []
                }
                
                for result in results:
                    # Convert result to dict (simplified)
                    result_dict = {
                        'success': result.success,
                        'document_type': result.document_type,
                        'extracted_fields': result.extracted_fields,
                        'validation_results': result.validation_results,
                        'confidence_score': result.confidence_score,
                        'processing_time': result.processing_time,
                        'quality_score': result.quality_score,
                        'metadata': result.metadata,
                        'errors': [{'message': e.message, 'severity': e.severity.value} for e in result.errors],
                        'warnings': [{'message': w.message, 'severity': w.severity.value} for w in result.warnings]
                    }
                    export_data['results'].append(result_dict)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Results exported to {output_path}")
                return True
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False