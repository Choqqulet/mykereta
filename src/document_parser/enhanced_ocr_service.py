#!/usr/bin/env python3
"""
Enhanced OCR Service Module

Advanced OCR service with multi-format support (PDF, JPG, PNG), quality assessment,
and intelligent preprocessing for Malaysian IC and Passport documents.
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import pytesseract
import easyocr
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import re
import fitz  # PyMuPDF for PDF processing
import io
# import magic  # Commented out to avoid dependency issues
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Image quality assessment metrics"""
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    noise_level: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    file_size: int = 0
    overall_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)

@dataclass
class OCRResult:
    """Enhanced OCR result with detailed metadata"""
    text: str = ""
    confidence: float = 0.0
    word_boxes: List[Dict] = field(default_factory=list)
    line_boxes: List[Dict] = field(default_factory=list)
    engine_used: str = ""
    processing_time: float = 0.0
    quality_metrics: Optional[QualityMetrics] = None
    preprocessing_applied: List[str] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class DocumentInfo:
    """Document format and metadata information"""
    format: str = ""
    mime_type: str = ""
    pages: int = 1
    dimensions: Tuple[int, int] = (0, 0)
    file_size: int = 0
    is_valid: bool = True
    security_check: bool = True

class EnhancedOCRService:
    """
    Enhanced OCR service with multi-format support and intelligent preprocessing.
    
    Features:
    - Multi-format support (PDF, JPG, PNG, TIFF, BMP)
    - Advanced image quality assessment and enhancement
    - Multi-engine OCR with consensus mechanism
    - Specialized preprocessing for IC and Passport documents
    - Security validation and format verification
    """
    
    def __init__(self, 
                 engines: List[str] = None,
                 languages: List[str] = None,
                 tesseract_path: Optional[str] = None,
                 enable_gpu: bool = False):
        """
        Initialize the enhanced OCR service.
        
        Args:
            engines: List of OCR engines ['tesseract', 'easyocr', 'paddleocr']
            languages: List of language codes ['en', 'ms', 'zh']
            tesseract_path: Path to tesseract executable
            enable_gpu: Enable GPU acceleration for supported engines
        """
        self.engines = engines or ['tesseract', 'easyocr']
        self.languages = languages or ['en', 'ms']
        self.enable_gpu = enable_gpu
        
        # Supported formats
        self.supported_formats = {
            'image': ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'],
            'document': ['.pdf']
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_sharpness': 100.0,
            'min_brightness': 50.0,
            'max_brightness': 200.0,
            'min_contrast': 30.0,
            'max_noise': 0.3,
            'min_resolution': (300, 300)
        }
        
        # Initialize OCR engines
        self._initialize_engines(tesseract_path)
        
        logger.info(f"Enhanced OCR Service initialized with engines: {self.engines}")
    
    def _initialize_engines(self, tesseract_path: Optional[str]):
        """Initialize OCR engines with error handling"""
        # Set tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Initialize EasyOCR
        self.easyocr_reader = None
        if 'easyocr' in self.engines:
            try:
                self.easyocr_reader = easyocr.Reader(
                    self.languages, 
                    gpu=self.enable_gpu
                )
                logger.info(f"EasyOCR initialized with languages: {self.languages}")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.engines = [eng for eng in self.engines if eng != 'easyocr']
        
        # Language mappings
        self.tesseract_lang_map = {
            'en': 'eng',
            'ms': 'msa',
            'zh': 'chi_sim'
        }
    
    def process_document(self, 
                        document_input: Union[str, bytes, Path],
                        document_type: str = "auto",
                        quality_enhancement: bool = True,
                        extract_coordinates: bool = False) -> Dict[str, Any]:
        """
        Process document with automatic format detection and enhancement.
        
        Args:
            document_input: Document file path, bytes, or Path object
            document_type: Document type hint ('ic', 'passport', 'auto')
            quality_enhancement: Apply quality enhancement preprocessing
            extract_coordinates: Extract text with coordinate information
            
        Returns:
            Dict: Complete processing results with metadata
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Document format detection and validation
            doc_info = self._detect_document_format(document_input)
            if not doc_info.is_valid:
                return self._create_error_result("Invalid document format")
            
            # Step 2: Extract images from document
            images = self._extract_images_from_document(document_input, doc_info)
            if not images:
                return self._create_error_result("No images extracted from document")
            
            # Step 3: Process each image/page
            page_results = []
            for i, image in enumerate(images):
                page_result = self._process_single_image(
                    image, 
                    document_type, 
                    quality_enhancement, 
                    extract_coordinates,
                    page_number=i+1
                )
                page_results.append(page_result)
            
            # Step 4: Combine results from all pages
            combined_result = self._combine_page_results(page_results)
            
            # Step 5: Add document metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "document_info": doc_info,
                "pages_processed": len(images),
                "combined_result": combined_result,
                "page_results": page_results,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return self._create_error_result(str(e))
    
    def _detect_document_format(self, document_input: Union[str, bytes, Path]) -> DocumentInfo:
        """Detect and validate document format"""
        try:
            if isinstance(document_input, (str, Path)):
                file_path = Path(document_input)
                if not file_path.exists():
                    return DocumentInfo(is_valid=False)
                
                # Get file info
                file_size = file_path.stat().st_size
                file_extension = file_path.suffix.lower()
                
                # Read file for MIME type detection
                with open(file_path, 'rb') as f:
                    file_bytes = f.read(1024)  # Read first 1KB for detection
                
            else:
                file_bytes = document_input[:1024] if len(document_input) > 1024 else document_input
                file_size = len(document_input)
                file_extension = ""
            
            # Detect MIME type using file extension fallback
            # mime_type = magic.from_buffer(file_bytes, mime=True)  # Commented out to avoid dependency
            mime_type = self._detect_mime_from_extension(file_extension)
            
            # Determine format
            if mime_type == 'application/pdf':
                format_type = 'pdf'
                pages = self._count_pdf_pages(document_input)
            elif mime_type.startswith('image/'):
                format_type = 'image'
                pages = 1
            else:
                return DocumentInfo(is_valid=False)
            
            # Security validation
            security_check = self._validate_document_security(file_bytes, mime_type)
            
            return DocumentInfo(
                format=format_type,
                mime_type=mime_type,
                pages=pages,
                file_size=file_size,
                is_valid=True,
                security_check=security_check
            )
            
        except Exception as e:
            logger.error(f"Format detection error: {e}")
            return DocumentInfo(is_valid=False)
    
    def _extract_images_from_document(self, 
                                    document_input: Union[str, bytes, Path], 
                                    doc_info: DocumentInfo) -> List[Image.Image]:
        """Extract images from various document formats"""
        images = []
        
        try:
            if doc_info.format == 'pdf':
                images = self._extract_images_from_pdf(document_input)
            elif doc_info.format == 'image':
                image = self._load_image(document_input)
                if image:
                    images = [image]
            
            return images
            
        except Exception as e:
            logger.error(f"Image extraction error: {e}")
            return []
    
    def _extract_images_from_pdf(self, pdf_input: Union[str, bytes, Path]) -> List[Image.Image]:
        """Extract images from PDF document"""
        images = []
        
        try:
            # Open PDF
            if isinstance(pdf_input, (str, Path)):
                pdf_document = fitz.open(pdf_input)
            else:
                pdf_document = fitz.open(stream=pdf_input, filetype="pdf")
            
            # Extract each page as image
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Convert page to image with high DPI for better OCR
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
            
            pdf_document.close()
            return images
            
        except Exception as e:
            logger.error(f"PDF image extraction error: {e}")
            return []
    
    def _process_single_image(self, 
                            image: Image.Image,
                            document_type: str,
                            quality_enhancement: bool,
                            extract_coordinates: bool,
                            page_number: int = 1) -> OCRResult:
        """Process a single image with OCR"""
        try:
            # Step 1: Quality assessment
            quality_metrics = self._assess_image_quality(image)
            
            # Step 2: Apply preprocessing based on quality and document type
            processed_image = image.copy()
            preprocessing_steps = []
            
            if quality_enhancement:
                processed_image, steps = self._enhance_image_quality(
                    processed_image, 
                    quality_metrics, 
                    document_type
                )
                preprocessing_steps.extend(steps)
            
            # Step 3: OCR extraction with multiple engines
            ocr_results = {}
            for engine in self.engines:
                try:
                    if engine == 'tesseract':
                        result = self._extract_with_tesseract(
                            processed_image, 
                            extract_coordinates
                        )
                    elif engine == 'easyocr' and self.easyocr_reader:
                        result = self._extract_with_easyocr(
                            processed_image, 
                            extract_coordinates
                        )
                    
                    if result:
                        ocr_results[engine] = result
                        
                except Exception as e:
                    logger.warning(f"OCR engine {engine} failed: {e}")
            
            # Step 4: Combine results using consensus
            final_result = self._combine_ocr_results(ocr_results)
            
            return OCRResult(
                text=final_result.get("text", ""),
                confidence=final_result.get("confidence", 0.0),
                word_boxes=final_result.get("word_boxes", []),
                line_boxes=final_result.get("line_boxes", []),
                engine_used=final_result.get("best_engine", ""),
                quality_metrics=quality_metrics,
                preprocessing_applied=preprocessing_steps
            )
            
        except Exception as e:
            logger.error(f"Single image processing error: {e}")
            return OCRResult(error=str(e))
    
    def _assess_image_quality(self, image: Image.Image) -> QualityMetrics:
        """Assess image quality for OCR optimization"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image.convert('L'))  # Convert to grayscale
            
            # Calculate sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate brightness (mean pixel value)
            brightness = np.mean(img_array)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(img_array)
            
            # Estimate noise level
            noise_level = self._estimate_noise(img_array)
            
            # Get resolution
            resolution = image.size
            
            # Calculate overall quality score
            overall_score = self._calculate_quality_score(
                sharpness, brightness, contrast, noise_level, resolution
            )
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                sharpness, brightness, contrast, noise_level, resolution
            )
            
            return QualityMetrics(
                sharpness=sharpness,
                brightness=brightness,
                contrast=contrast,
                noise_level=noise_level,
                resolution=resolution,
                overall_score=overall_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return QualityMetrics()
    
    def _enhance_image_quality(self, 
                             image: Image.Image, 
                             quality_metrics: QualityMetrics,
                             document_type: str) -> Tuple[Image.Image, List[str]]:
        """Enhance image quality based on assessment and document type"""
        enhanced_image = image.copy()
        applied_steps = []
        
        try:
            # Convert to RGB if needed
            if enhanced_image.mode != 'RGB':
                enhanced_image = enhanced_image.convert('RGB')
                applied_steps.append("color_mode_conversion")
            
            # Apply document-specific preprocessing
            if document_type in ['ic', 'passport']:
                enhanced_image = self._apply_document_specific_enhancement(
                    enhanced_image, document_type
                )
                applied_steps.append(f"{document_type}_specific_enhancement")
            
            # Brightness adjustment
            if quality_metrics.brightness < self.quality_thresholds['min_brightness']:
                enhancer = ImageEnhance.Brightness(enhanced_image)
                factor = self.quality_thresholds['min_brightness'] / quality_metrics.brightness
                enhanced_image = enhancer.enhance(min(factor, 2.0))
                applied_steps.append("brightness_enhancement")
            elif quality_metrics.brightness > self.quality_thresholds['max_brightness']:
                enhancer = ImageEnhance.Brightness(enhanced_image)
                factor = self.quality_thresholds['max_brightness'] / quality_metrics.brightness
                enhanced_image = enhancer.enhance(max(factor, 0.5))
                applied_steps.append("brightness_reduction")
            
            # Contrast adjustment
            if quality_metrics.contrast < self.quality_thresholds['min_contrast']:
                enhancer = ImageEnhance.Contrast(enhanced_image)
                enhanced_image = enhancer.enhance(1.5)
                applied_steps.append("contrast_enhancement")
            
            # Sharpness enhancement
            if quality_metrics.sharpness < self.quality_thresholds['min_sharpness']:
                enhancer = ImageEnhance.Sharpness(enhanced_image)
                enhanced_image = enhancer.enhance(1.3)
                applied_steps.append("sharpness_enhancement")
            
            # Noise reduction
            if quality_metrics.noise_level > self.quality_thresholds['max_noise']:
                enhanced_image = enhanced_image.filter(ImageFilter.MedianFilter(size=3))
                applied_steps.append("noise_reduction")
            
            # Resolution enhancement if needed
            if (quality_metrics.resolution[0] < self.quality_thresholds['min_resolution'][0] or 
                quality_metrics.resolution[1] < self.quality_thresholds['min_resolution'][1]):
                
                scale_factor = max(
                    self.quality_thresholds['min_resolution'][0] / quality_metrics.resolution[0],
                    self.quality_thresholds['min_resolution'][1] / quality_metrics.resolution[1]
                )
                
                new_size = (
                    int(quality_metrics.resolution[0] * scale_factor),
                    int(quality_metrics.resolution[1] * scale_factor)
                )
                
                enhanced_image = enhanced_image.resize(new_size, Image.Resampling.LANCZOS)
                applied_steps.append("resolution_upscaling")
            
            return enhanced_image, applied_steps
            
        except Exception as e:
            logger.error(f"Image enhancement error: {e}")
            return image, applied_steps
    
    def _apply_document_specific_enhancement(self, 
                                           image: Image.Image, 
                                           document_type: str) -> Image.Image:
        """Apply document-specific enhancement techniques"""
        try:
            if document_type == 'ic':
                # Malaysian IC specific enhancements
                # - Enhance blue/red text contrast
                # - Improve hologram area readability
                enhanced = self._enhance_ic_document(image)
            elif document_type == 'passport':
                # Passport specific enhancements
                # - Enhance MRZ (Machine Readable Zone) area
                # - Improve photo and text contrast
                enhanced = self._enhance_passport_document(image)
            else:
                enhanced = image
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Document-specific enhancement error: {e}")
            return image
    
    def _enhance_ic_document(self, image: Image.Image) -> Image.Image:
        """Enhance Malaysian IC document for better OCR"""
        try:
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Enhance blue text (common in Malaysian IC)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Define blue color range
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Create mask for blue areas
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Enhance blue areas
            img_array[blue_mask > 0] = [0, 0, 0]  # Convert blue text to black
            
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.error(f"IC enhancement error: {e}")
            return image
    
    def _enhance_passport_document(self, image: Image.Image) -> Image.Image:
        """Enhance passport document for better OCR"""
        try:
            # Focus on MRZ area enhancement
            img_array = np.array(image.convert('L'))
            
            # Apply adaptive thresholding for better text contrast
            adaptive_thresh = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return Image.fromarray(adaptive_thresh).convert('RGB')
            
        except Exception as e:
            logger.error(f"Passport enhancement error: {e}")
            return image
    
    def _extract_with_tesseract(self, 
                              image: Image.Image, 
                              extract_coordinates: bool = False) -> Dict:
        """Extract text using Tesseract OCR"""
        try:
            # Prepare tesseract languages
            tesseract_langs = '+'.join([
                self.tesseract_lang_map.get(lang, lang) 
                for lang in self.languages
            ])
            
            # Configure tesseract
            config = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/- '
            
            if extract_coordinates:
                # Extract with bounding boxes
                data = pytesseract.image_to_data(
                    image, 
                    lang=tesseract_langs,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Process coordinate data
                word_boxes = []
                line_boxes = []
                text_parts = []
                
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 30:  # Confidence threshold
                        word_info = {
                            'text': data['text'][i],
                            'confidence': int(data['conf'][i]),
                            'bbox': [
                                data['left'][i],
                                data['top'][i],
                                data['left'][i] + data['width'][i],
                                data['top'][i] + data['height'][i]
                            ]
                        }
                        word_boxes.append(word_info)
                        text_parts.append(data['text'][i])
                
                text = ' '.join(text_parts)
                avg_confidence = np.mean([int(conf) for conf in data['conf'] if int(conf) > 0])
                
            else:
                # Simple text extraction
                text = pytesseract.image_to_string(
                    image, 
                    lang=tesseract_langs,
                    config=config
                )
                
                # Get confidence
                data = pytesseract.image_to_data(
                    image, 
                    lang=tesseract_langs,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0
                
                word_boxes = []
                line_boxes = []
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence / 100.0,  # Convert to 0-1 scale
                'word_boxes': word_boxes,
                'line_boxes': line_boxes,
                'engine': 'tesseract'
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction error: {e}")
            return {'text': '', 'confidence': 0.0, 'error': str(e)}
    
    def _extract_with_easyocr(self, 
                            image: Image.Image, 
                            extract_coordinates: bool = False) -> Dict:
        """Extract text using EasyOCR"""
        try:
            if not self.easyocr_reader:
                return {'text': '', 'confidence': 0.0, 'error': 'EasyOCR not available'}
            
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Extract text
            results = self.easyocr_reader.readtext(img_array)
            
            # Process results
            text_parts = []
            word_boxes = []
            total_confidence = 0
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Confidence threshold
                    text_parts.append(text)
                    total_confidence += confidence
                    
                    if extract_coordinates:
                        # Convert bbox to standard format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        
                        word_info = {
                            'text': text,
                            'confidence': confidence,
                            'bbox': [
                                min(x_coords),
                                min(y_coords),
                                max(x_coords),
                                max(y_coords)
                            ]
                        }
                        word_boxes.append(word_info)
            
            text = ' '.join(text_parts)
            avg_confidence = total_confidence / len(results) if results else 0
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'word_boxes': word_boxes,
                'line_boxes': [],
                'engine': 'easyocr'
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction error: {e}")
            return {'text': '', 'confidence': 0.0, 'error': str(e)}
    
    def _combine_ocr_results(self, results: Dict) -> Dict:
        """Combine results from multiple OCR engines using consensus"""
        if not results:
            return {'text': '', 'confidence': 0.0, 'best_engine': ''}
        
        if len(results) == 1:
            engine_name = list(results.keys())[0]
            result = results[engine_name]
            result['best_engine'] = engine_name
            return result
        
        # Weighted consensus based on confidence scores
        best_result = None
        best_score = 0
        best_engine = ""
        
        for engine, result in results.items():
            # Calculate weighted score (confidence * text length)
            score = result.get('confidence', 0) * len(result.get('text', ''))
            
            if score > best_score:
                best_score = score
                best_result = result
                best_engine = engine
        
        if best_result:
            best_result['best_engine'] = best_engine
            return best_result
        
        return {'text': '', 'confidence': 0.0, 'best_engine': ''}
    
    def _combine_page_results(self, page_results: List[OCRResult]) -> OCRResult:
        """Combine results from multiple pages"""
        if not page_results:
            return OCRResult()
        
        if len(page_results) == 1:
            return page_results[0]
        
        # Combine text from all pages
        combined_text = []
        total_confidence = 0
        all_word_boxes = []
        all_preprocessing = []
        
        for i, result in enumerate(page_results):
            if result.text:
                combined_text.append(f"[Page {i+1}]\n{result.text}")
                total_confidence += result.confidence
                all_word_boxes.extend(result.word_boxes)
                all_preprocessing.extend(result.preprocessing_applied)
        
        avg_confidence = total_confidence / len(page_results) if page_results else 0
        
        return OCRResult(
            text='\n\n'.join(combined_text),
            confidence=avg_confidence,
            word_boxes=all_word_boxes,
            preprocessing_applied=list(set(all_preprocessing))
        )
    
    # Helper methods
    def _load_image(self, image_input: Union[str, bytes, Path]) -> Optional[Image.Image]:
        """Load image from various input types"""
        try:
            if isinstance(image_input, (str, Path)):
                return Image.open(image_input)
            elif isinstance(image_input, bytes):
                return Image.open(io.BytesIO(image_input))
            else:
                return None
        except Exception as e:
            logger.error(f"Image loading error: {e}")
            return None
    
    def _count_pdf_pages(self, pdf_input: Union[str, bytes, Path]) -> int:
        """Count pages in PDF document"""
        try:
            if isinstance(pdf_input, (str, Path)):
                pdf_document = fitz.open(pdf_input)
            else:
                pdf_document = fitz.open(stream=pdf_input, filetype="pdf")
            
            page_count = pdf_document.page_count
            pdf_document.close()
            return page_count
        except Exception:
            return 0
    
    def _validate_document_security(self, file_bytes: bytes, mime_type: str) -> bool:
        """Basic security validation for uploaded documents"""
        try:
            # Check file size (max 50MB)
            if len(file_bytes) > 50 * 1024 * 1024:
                return False
            
            # Check for malicious patterns (basic check)
            malicious_patterns = [b'<script', b'javascript:', b'vbscript:']
            for pattern in malicious_patterns:
                if pattern in file_bytes.lower():
                    return False
            
            return True
        except Exception:
            return False
    
    def _estimate_noise(self, img_array: np.ndarray) -> float:
        """Estimate noise level in image"""
        try:
            # Use Laplacian to estimate noise
            laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
            noise_level = np.var(laplacian) / np.mean(img_array)
            return min(noise_level / 1000.0, 1.0)  # Normalize to 0-1
        except Exception:
            return 0.0
    
    def _calculate_quality_score(self, 
                               sharpness: float, 
                               brightness: float, 
                               contrast: float, 
                               noise_level: float, 
                               resolution: Tuple[int, int]) -> float:
        """Calculate overall quality score (0-100)"""
        try:
            # Normalize metrics
            sharpness_score = min(sharpness / 200.0, 1.0) * 25
            brightness_score = (1.0 - abs(brightness - 128) / 128.0) * 25
            contrast_score = min(contrast / 50.0, 1.0) * 25
            noise_score = (1.0 - noise_level) * 15
            resolution_score = min(min(resolution) / 300.0, 1.0) * 10
            
            total_score = (sharpness_score + brightness_score + 
                          contrast_score + noise_score + resolution_score)
            
            return min(total_score, 100.0)
        except Exception:
            return 0.0
    
    def _generate_quality_recommendations(self, 
                                        sharpness: float, 
                                        brightness: float, 
                                        contrast: float, 
                                        noise_level: float, 
                                        resolution: Tuple[int, int]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if sharpness < self.quality_thresholds['min_sharpness']:
            recommendations.append("Image appears blurry - consider using a tripod or better lighting")
        
        if brightness < self.quality_thresholds['min_brightness']:
            recommendations.append("Image is too dark - increase lighting or exposure")
        elif brightness > self.quality_thresholds['max_brightness']:
            recommendations.append("Image is too bright - reduce lighting or exposure")
        
        if contrast < self.quality_thresholds['min_contrast']:
            recommendations.append("Low contrast - ensure good lighting and avoid shadows")
        
        if noise_level > self.quality_thresholds['max_noise']:
            recommendations.append("High noise level - use better lighting and lower ISO")
        
        if (resolution[0] < self.quality_thresholds['min_resolution'][0] or 
            resolution[1] < self.quality_thresholds['min_resolution'][1]):
            recommendations.append("Low resolution - use higher quality camera settings")
        
        return recommendations
    
    def _detect_mime_from_extension(self, extension: str) -> str:
        """Detect MIME type from file extension."""
        ext = extension.lower()
        type_map = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.bmp': 'image/bmp'
        }
        return type_map.get(ext, 'application/octet-stream')
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "success": False,
            "error": error_message,
            "document_info": None,
            "pages_processed": 0,
            "combined_result": OCRResult(error=error_message),
            "page_results": [],
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }