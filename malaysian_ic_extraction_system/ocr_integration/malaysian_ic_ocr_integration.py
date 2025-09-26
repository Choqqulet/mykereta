#!/usr/bin/env python3
"""
Malaysian IC OCR Integration Module

This module integrates the successful techniques from the Malaysian IC OCR repository
into our enhanced document parser, specifically:
1. CRAFT text detection for precise text localization
2. Advanced preprocessing techniques optimized for Malaysian IC documents
3. Dual OCR approach with Tesseract and EasyOCR
4. Improved thresholding and image enhancement

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import easyocr
import pytesseract
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
import time
from pathlib import Path
import torch
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MalaysianICOCRResult:
    """Result structure for Malaysian IC OCR processing"""
    success: bool = False
    text: str = ""
    combined_text: str = ""
    confidence: float = 0.0
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    detected_fields: Dict[str, str] = field(default_factory=dict)
    extracted_fields: Dict[str, str] = field(default_factory=dict)
    processing_time: float = 0.0
    engine_used: str = ""
    preprocessing_applied: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    tesseract_result: Dict[str, Any] = field(default_factory=dict)
    easyocr_result: Dict[str, Any] = field(default_factory=dict)

class MalaysianICPreprocessor:
    """
    Advanced preprocessing techniques inspired by Malaysian IC OCR repository
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply_malaysian_ic_preprocessing(self, image: Union[Image.Image, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Apply Malaysian IC specific preprocessing techniques
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Tuple of (processed_image, applied_techniques)
        """
        start_time = time.time()
        applied_techniques = []
        
        # Convert PIL to OpenCV if needed
        if isinstance(image, Image.Image):
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_array = image.copy()
        
        # 1. Grayscale conversion (Malaysian IC OCR technique)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            applied_techniques.append("grayscale_conversion")
        else:
            gray = img_array.copy()
        
        # 2. Advanced thresholding (inspired by their THRESH_TRUNC approach)
        enhanced_gray = self._apply_malaysian_thresholding(gray)
        applied_techniques.append("malaysian_thresholding")
        
        # 3. Noise reduction
        denoised = cv2.medianBlur(enhanced_gray, 3)
        applied_techniques.append("noise_reduction")
        
        # 4. Contrast enhancement
        enhanced = self._enhance_contrast_for_ic(denoised)
        applied_techniques.append("contrast_enhancement")
        
        # 5. Morphological operations for text clarity
        morphed = self._apply_morphological_operations(enhanced)
        applied_techniques.append("morphological_operations")
        
        processing_time = time.time() - start_time
        self.logger.info(f"Malaysian IC preprocessing completed in {processing_time:.3f}s")
        
        return morphed, applied_techniques
    
    def _apply_malaysian_thresholding(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Apply the specific thresholding technique used in Malaysian IC OCR
        Based on: cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)[1]
        """
        # Primary thresholding (Malaysian IC OCR approach)
        _, thresh_trunc = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TRUNC)
        
        # Additional adaptive thresholding for better results
        adaptive_thresh = cv2.adaptiveThreshold(
            thresh_trunc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Combine both techniques
        combined = cv2.bitwise_and(thresh_trunc, adaptive_thresh)
        
        return combined
    
    def _enhance_contrast_for_ic(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast specifically for Malaysian IC documents
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def _apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to improve text clarity
        """
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Opening to remove noise
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed

class CRAFTTextDetector:
    """
    CRAFT (Character Region Awareness for Text Detection) implementation
    Inspired by the Malaysian IC OCR repository's approach
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # CRAFT parameters (from Malaysian IC OCR)
        self.text_threshold = 0.7
        self.low_text = 0.4
        self.link_threshold = 0.4
        self.canvas_size = 2560
        self.mag_ratio = 1.0
        
    def detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using CRAFT-like approach
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2)
        """
        # For now, implement a simplified version
        # In production, you would use the actual CRAFT model
        return self._simplified_text_detection(image)
    
    def _simplified_text_detection(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Simplified text detection using OpenCV contours
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours and create bounding boxes
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size (typical text characteristics)
            if w > 10 and h > 10 and w < image.shape[1] * 0.8 and h < image.shape[0] * 0.3:
                text_regions.append((x, y, x + w, y + h))
        
        return text_regions

class MalaysianICOCREngine:
    """
    Enhanced OCR engine incorporating Malaysian IC OCR techniques
    """
    
    def __init__(self, 
                 engines: List[str] = None,
                 languages: List[str] = None,
                 enable_gpu: bool = False):
        """
        Initialize the Malaysian IC OCR engine
        
        Args:
            engines: List of OCR engines to use ['tesseract', 'easyocr']
            languages: List of languages ['en', 'ms']
            enable_gpu: Enable GPU acceleration for EasyOCR
        """
        self.engines = engines or ['tesseract', 'easyocr']
        self.languages = languages or ['en', 'ms']
        self.enable_gpu = enable_gpu
        
        self.preprocessor = MalaysianICPreprocessor()
        self.text_detector = CRAFTTextDetector()
        
        # Initialize EasyOCR (Malaysian IC OCR approach)
        self.easyocr_reader = None
        if 'easyocr' in self.engines:
            try:
                self.easyocr_reader = easyocr.Reader(self.languages, gpu=self.enable_gpu)
                logger.info(f"EasyOCR initialized with languages: {self.languages}")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.engines = [eng for eng in self.engines if eng != 'easyocr']
        
        logger.info(f"Malaysian IC OCR Engine initialized with engines: {self.engines}")
    
    def process_malaysian_ic(self, 
                           image_input: Union[str, Image.Image, np.ndarray],
                           use_preprocessing: bool = True,
                           extract_coordinates: bool = True) -> MalaysianICOCRResult:
        """
        Process Malaysian IC document using integrated techniques
        
        Args:
            image_input: Input image (path, PIL Image, or numpy array)
            use_preprocessing: Apply Malaysian IC specific preprocessing
            extract_coordinates: Extract text coordinates
            
        Returns:
            MalaysianICOCRResult with extracted text and metadata
        """
        start_time = time.time()
        
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        
        # Apply preprocessing if enabled
        preprocessing_applied = []
        if use_preprocessing:
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_img, techniques = self.preprocessor.apply_malaysian_ic_preprocessing(img_array)
            preprocessing_applied.extend(techniques)
            
            # Convert back to PIL for OCR
            processed_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        else:
            processed_pil = image
        
        # Dual OCR approach (Malaysian IC OCR technique)
        ocr_results = {}
        
        # Tesseract OCR
        if 'tesseract' in self.engines:
            tesseract_result = self._extract_with_tesseract(processed_pil, extract_coordinates)
            ocr_results['tesseract'] = tesseract_result
        
        # EasyOCR (CRAFT + CRNN approach)
        if 'easyocr' in self.engines and self.easyocr_reader:
            easyocr_result = self._extract_with_easyocr(processed_pil, extract_coordinates)
            ocr_results['easyocr'] = easyocr_result
        
        # Combine results using consensus
        final_result = self._combine_ocr_results(ocr_results)
        
        # Extract Malaysian IC specific fields
        detected_fields = self._extract_ic_fields(final_result['text'])
        
        processing_time = time.time() - start_time
        
        # Determine success based on confidence and text extraction
        success = final_result['confidence'] > 0.3 and len(final_result['text'].strip()) > 0
        
        return MalaysianICOCRResult(
            success=success,
            text=final_result['text'],
            combined_text=final_result['text'],  # Same as text for now
            confidence=final_result['confidence'],
            bounding_boxes=final_result.get('bounding_boxes', []),
            detected_fields=detected_fields,
            extracted_fields=detected_fields,  # Same as detected_fields
            processing_time=processing_time,
            engine_used=final_result['engine_used'],
            preprocessing_applied=preprocessing_applied,
            quality_score=final_result.get('quality_score', 0.0),
            tesseract_result=ocr_results.get('tesseract', {}),
            easyocr_result=ocr_results.get('easyocr', {})
        )
    
    def _extract_with_tesseract(self, image: Image.Image, extract_coordinates: bool) -> Dict:
        """
        Extract text using Tesseract with Malaysian IC optimizations
        """
        try:
            # Configure Tesseract for Malaysian IC
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@.-/ '
            
            # Extract text
            text = pytesseract.image_to_string(image, config=config, lang='eng+msa')
            
            # Extract coordinates if requested
            bounding_boxes = []
            if extract_coordinates:
                data = pytesseract.image_to_data(image, config=config, lang='eng+msa', output_type=pytesseract.Output.DICT)
                for i, word in enumerate(data['text']):
                    if word.strip():
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bounding_boxes.append((x, y, x + w, y + h))
            
            # Calculate confidence
            confidences = pytesseract.image_to_data(image, config=config, lang='eng+msa', output_type=pytesseract.Output.DICT)['conf']
            avg_confidence = np.mean([c for c in confidences if c > 0]) if confidences else 0
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence / 100.0,
                'bounding_boxes': bounding_boxes,
                'engine': 'tesseract'
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {'text': '', 'confidence': 0.0, 'bounding_boxes': [], 'engine': 'tesseract'}
    
    def _extract_with_easyocr(self, image: Image.Image, extract_coordinates: bool) -> Dict:
        """
        Extract text using EasyOCR (CRAFT + CRNN approach from Malaysian IC OCR)
        """
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Use EasyOCR with detailed output (Malaysian IC OCR approach)
            results = self.easyocr_reader.readtext(img_array, detail=1)
            
            # Extract text and coordinates
            text_parts = []
            bounding_boxes = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if text.strip() and confidence > 0.3:  # Filter low confidence
                    text_parts.append(text)
                    confidences.append(confidence)
                    
                    if extract_coordinates:
                        # Convert EasyOCR bbox format to our format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x1, y1 = int(min(x_coords)), int(min(y_coords))
                        x2, y2 = int(max(x_coords)), int(max(y_coords))
                        bounding_boxes.append((x1, y1, x2, y2))
            
            # Combine text
            combined_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'bounding_boxes': bounding_boxes,
                'engine': 'easyocr'
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {'text': '', 'confidence': 0.0, 'bounding_boxes': [], 'engine': 'easyocr'}
    
    def _combine_ocr_results(self, results: Dict) -> Dict:
        """
        Combine OCR results using consensus approach
        """
        if not results:
            return {'text': '', 'confidence': 0.0, 'engine_used': 'none'}
        
        # If only one engine, return its result
        if len(results) == 1:
            engine_name = list(results.keys())[0]
            result = results[engine_name]
            result['engine_used'] = engine_name
            return result
        
        # Consensus approach: choose result with higher confidence
        best_result = None
        best_confidence = 0.0
        best_engine = 'none'
        
        for engine, result in results.items():
            if result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_result = result
                best_engine = engine
        
        if best_result:
            best_result['engine_used'] = best_engine
            return best_result
        
        return {'text': '', 'confidence': 0.0, 'engine_used': 'none'}
    
    def _extract_ic_fields(self, text: str) -> Dict[str, str]:
        """
        Extract specific Malaysian IC fields from OCR text
        """
        fields = {}
        
        # IC Number pattern (YYMMDD-PB-XXXX)
        ic_pattern = r'\b\d{6}-\d{2}-\d{4}\b'
        ic_match = re.search(ic_pattern, text)
        if ic_match:
            fields['ic_number'] = ic_match.group()
        
        # Name extraction (simplified)
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 3 and not re.search(r'\d{6}-\d{2}-\d{4}', line):
                # Potential name line
                if 'name' not in fields and re.match(r'^[A-Za-z\s]+$', line):
                    fields['name'] = line
        
        return fields

# Integration function for existing enhanced document processor
def integrate_malaysian_ic_techniques(enhanced_ocr_service):
    """
    Integrate Malaysian IC OCR techniques into existing enhanced OCR service
    """
    # Add Malaysian IC OCR engine as an additional capability
    malaysian_ic_engine = MalaysianICOCREngine()
    
    # Monkey patch the enhanced OCR service to include Malaysian IC processing
    def process_malaysian_ic_document(self, image_input, **kwargs):
        return malaysian_ic_engine.process_malaysian_ic(image_input, **kwargs)
    
    # Add the method to the enhanced OCR service
    enhanced_ocr_service.process_malaysian_ic_document = process_malaysian_ic_document.__get__(enhanced_ocr_service)
    
    logger.info("Malaysian IC OCR techniques integrated successfully")
    
    return enhanced_ocr_service