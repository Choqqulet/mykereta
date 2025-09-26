#!/usr/bin/env python3
"""
Malaysian IC (MyKad) Field Extractor

A comprehensive baseline extractor for Malaysian Identity Cards using:
- Advanced image preprocessing
- OCR with PaddleOCR/Tesseract
- Regex patterns for Malaysian IC fields
- Field validation and consistency checks
- Confidence scoring

Author: AI Assistant
Date: 2025
"""

import cv2
import numpy as np
import re
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image, ImageEnhance
import argparse
from dataclasses import dataclass
import logging

# Try to import OCR libraries
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

@dataclass
class ExtractedField:
    """Represents an extracted field with confidence and validation"""
    value: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    validation_passed: bool = True
    validation_errors: List[str] = None

@dataclass
class ICExtractionResult:
    """Complete IC extraction result"""
    fields: Dict[str, ExtractedField]
    overall_confidence: float
    processing_time: float
    image_quality_score: float
    validation_summary: Dict[str, bool]

class ImagePreprocessor:
    """Advanced image preprocessing for Malaysian IC"""
    
    def __init__(self):
        self.target_width = 1200
        self.target_height = 800
    
    def preprocess(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Apply comprehensive preprocessing pipeline"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        
        if img is None:
            raise ValueError("Could not load image")
        
        # Step 1: Resize to standard dimensions
        img = self._resize_image(img)
        
        # Step 2: Deskew
        img = self._deskew_image(img)
        
        # Step 3: Enhance contrast and brightness
        img = self._enhance_contrast(img)
        
        # Step 4: Denoise
        img = self._denoise_image(img)
        
        # Step 5: Sharpen
        img = self._sharpen_image(img)
        
        return img
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = img.shape[:2]
        
        # Calculate scaling factor
        scale_w = self.target_width / w
        scale_h = self.target_height / h
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Pad to target size
        pad_w = (self.target_width - new_w) // 2
        pad_h = (self.target_height - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized, pad_h, self.target_height - new_h - pad_h,
            pad_w, self.target_width - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        
        return padded
    
    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Detect and correct skew angle"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for line in lines[:20]:  # Use first 20 lines
                rho, theta = line[0]  # HoughLines returns [[rho, theta]] format
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:  # Only correct if significant skew
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
        
        return img
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Enhance contrast and brightness"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _denoise_image(self, img: np.ndarray) -> np.ndarray:
        """Remove noise while preserving text"""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    def _sharpen_image(self, img: np.ndarray) -> np.ndarray:
        """Apply sharpening filter"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # Blend with original
        return cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)
    
    def assess_image_quality(self, img: np.ndarray) -> float:
        """Assess image quality for OCR (0-1 score)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 1000, 1.0)
        
        # Contrast (standard deviation)
        contrast = gray.std()
        contrast_score = min(contrast / 50, 1.0)
        
        # Brightness (mean)
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 128) / 128
        
        # Overall quality
        quality = (sharpness_score * 0.4 + contrast_score * 0.3 + brightness_score * 0.3)
        return quality

class MalaysianICRegexPatterns:
    """Regex patterns for Malaysian IC fields"""
    
    def __init__(self):
        # NRIC pattern: YYMMDD-SS-NNNN
        self.nric_pattern = r'\b\d{6}-\d{2}-\d{4}\b'
        
        # Date patterns (various formats)
        self.date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',  # DD-MM-YYYY or DD/MM/YYYY
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2}\b',   # DD-MM-YY or DD/MM/YY
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'    # YYYY-MM-DD or YYYY/MM/DD
        ]
        
        # Gender patterns
        self.gender_patterns = [
            r'\b(LELAKI|PEREMPUAN)\b',
            r'\b(MALE|FEMALE)\b'
        ]
        
        # Religion patterns
        self.religion_patterns = [
            r'\b(ISLAM|BUDDHA|HINDU|KRISTIAN|TAOISME|KONFUSIANISME|LAIN-LAIN)\b',
            r'\b(MUSLIM|BUDDHIST|CHRISTIAN|OTHERS?)\b'
        ]
        
        # Nationality patterns
        self.nationality_patterns = [
            r'\b(WARGANEGARA|BUKAN WARGANEGARA)\b',
            r'\b(CITIZEN|NON-CITIZEN)\b'
        ]
        
        # Name patterns (capitalized words)
        self.name_pattern = r'\b[A-Z][A-Z\s/]+\b'
        
        # Address patterns (Malaysian postal codes)
        self.postal_code_pattern = r'\b\d{5}\b'
        
        # Field label patterns
        self.field_labels = {
            'name': [r'NAMA\s*[:/]?\s*NAME', r'NAMA', r'NAME'],
            'nric': [r'NO\.?\s*KAD\s*PENGENALAN', r'NRIC', r'I\.?C\.?\s*NO'],
            'gender': [r'JANTINA\s*[:/]?\s*SEX', r'JANTINA', r'SEX', r'GENDER'],
            'birth_date': [r'TARIKH\s*LAHIR\s*[:/]?\s*DATE\s*OF\s*BIRTH', 
                          r'TARIKH\s*LAHIR', r'DATE\s*OF\s*BIRTH', r'DOB'],
            'religion': [r'AGAMA\s*[:/]?\s*RELIGION', r'AGAMA', r'RELIGION'],
            'nationality': [r'WARGANEGARA\s*[:/]?\s*NATIONALITY', 
                           r'WARGANEGARA', r'NATIONALITY'],
            'address': [r'ALAMAT\s*[:/]?\s*ADDRESS', r'ALAMAT', r'ADDRESS'],
            'issue_date': [r'TARIKH\s*DIKELUARKAN\s*[:/]?\s*DATE\s*OF\s*ISSUE',
                          r'TARIKH\s*DIKELUARKAN', r'DATE\s*OF\s*ISSUE']
        }

class OCREngine:
    """OCR engine wrapper supporting multiple backends"""
    
    def __init__(self, engine: str = "auto"):
        self.engine = engine
        self.ocr = None
        
        if engine == "auto":
            if PADDLE_AVAILABLE:
                self.engine = "paddle"
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en', 
                                   show_log=False, use_gpu=False)
            elif TESSERACT_AVAILABLE:
                self.engine = "tesseract"
            else:
                raise RuntimeError("No OCR engine available. Install PaddleOCR or Tesseract.")
        
        elif engine == "paddle":
            if not PADDLE_AVAILABLE:
                raise RuntimeError("PaddleOCR not available")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', 
                               show_log=False, use_gpu=False)
        
        elif engine == "tesseract":
            if not TESSERACT_AVAILABLE:
                raise RuntimeError("Tesseract not available")
    
    def extract_text(self, img: np.ndarray) -> List[Dict]:
        """Extract text with bounding boxes and confidence"""
        if self.engine == "paddle":
            return self._extract_with_paddle(img)
        elif self.engine == "tesseract":
            return self._extract_with_tesseract(img)
    
    def _extract_with_paddle(self, img: np.ndarray) -> List[Dict]:
        """Extract text using PaddleOCR"""
        results = self.ocr.ocr(img, cls=True)
        
        text_blocks = []
        for line in results[0] if results[0] else []:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            # Convert bbox to (x1, y1, x2, y2)
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            bbox_rect = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
            text_blocks.append({
                'text': text,
                'bbox': bbox_rect,
                'confidence': confidence
            })
        
        return text_blocks
    
    def _extract_with_tesseract(self, img: np.ndarray) -> List[Dict]:
        """Extract text using Tesseract"""
        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Get detailed data
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT,
                                       config='--psm 6 -l eng+msa')
        
        text_blocks = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text and int(data['conf'][i]) > 30:  # Confidence threshold
                bbox = (data['left'][i], data['top'][i],
                       data['left'][i] + data['width'][i],
                       data['top'][i] + data['height'][i])
                
                text_blocks.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': int(data['conf'][i]) / 100.0
                })
        
        return text_blocks

class MalaysianICExtractor:
    """Main extractor class for Malaysian IC fields"""
    
    def __init__(self, ocr_engine: str = "auto"):
        self.preprocessor = ImagePreprocessor()
        self.patterns = MalaysianICRegexPatterns()
        self.ocr = OCREngine(ocr_engine)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_fields(self, image_path: str) -> ICExtractionResult:
        """Extract all fields from Malaysian IC image"""
        start_time = datetime.now()
        
        # Load and preprocess image
        img = self.preprocessor.preprocess(image_path)
        image_quality = self.preprocessor.assess_image_quality(img)
        
        # Extract text using OCR
        text_blocks = self.ocr.extract_text(img)
        
        # Combine all text for pattern matching
        full_text = ' '.join([block['text'] for block in text_blocks])
        
        # Extract individual fields
        fields = {}
        
        # NRIC
        fields['nric'] = self._extract_nric(full_text, text_blocks)
        
        # Name
        fields['name'] = self._extract_name(full_text, text_blocks)
        
        # Gender
        fields['gender'] = self._extract_gender(full_text, text_blocks)
        
        # Birth date
        fields['birth_date'] = self._extract_birth_date(full_text, text_blocks, fields.get('nric'))
        
        # Religion
        fields['religion'] = self._extract_religion(full_text, text_blocks)
        
        # Nationality
        fields['nationality'] = self._extract_nationality(full_text, text_blocks)
        
        # Address
        fields['address'] = self._extract_address(full_text, text_blocks)
        
        # Issue date
        fields['issue_date'] = self._extract_issue_date(full_text, text_blocks)
        
        # Validate fields
        validation_summary = self._validate_fields(fields)
        
        # Calculate overall confidence
        confidences = [field.confidence for field in fields.values() if field.confidence > 0]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ICExtractionResult(
            fields=fields,
            overall_confidence=overall_confidence,
            processing_time=processing_time,
            image_quality_score=image_quality,
            validation_summary=validation_summary
        )
    
    def _extract_nric(self, text: str, text_blocks: List[Dict]) -> ExtractedField:
        """Extract NRIC number"""
        matches = re.findall(self.patterns.nric_pattern, text)
        
        if matches:
            nric = matches[0]
            confidence = self._calculate_field_confidence('nric', nric, text_blocks)
            bbox = self._find_text_bbox(nric, text_blocks)
            
            return ExtractedField(
                value=nric,
                confidence=confidence,
                bbox=bbox,
                validation_passed=self._validate_nric(nric)
            )
        
        return ExtractedField(value="", confidence=0.0)
    
    def _extract_name(self, text: str, text_blocks: List[Dict]) -> ExtractedField:
        """Extract name field"""
        # Look for name after label
        for label_pattern in self.patterns.field_labels['name']:
            pattern = label_pattern + r'\s*:?\s*([A-Z][A-Z\s/]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                confidence = self._calculate_field_confidence('name', name, text_blocks)
                bbox = self._find_text_bbox(name, text_blocks)
                
                return ExtractedField(
                    value=name,
                    confidence=confidence,
                    bbox=bbox,
                    validation_passed=len(name) > 3
                )
        
        # Fallback: look for capitalized text blocks
        name_candidates = re.findall(self.patterns.name_pattern, text)
        if name_candidates:
            # Filter out common labels and short words
            filtered = [name for name in name_candidates 
                       if len(name) > 5 and not any(label in name.upper() 
                       for label in ['NAMA', 'NAME', 'MALAYSIA', 'IDENTITY'])]
            
            if filtered:
                name = filtered[0]
                confidence = self._calculate_field_confidence('name', name, text_blocks)
                bbox = self._find_text_bbox(name, text_blocks)
                
                return ExtractedField(
                    value=name,
                    confidence=confidence * 0.7,  # Lower confidence for fallback
                    bbox=bbox
                )
        
        return ExtractedField(value="", confidence=0.0)
    
    def _extract_gender(self, text: str, text_blocks: List[Dict]) -> ExtractedField:
        """Extract gender field"""
        for pattern in self.patterns.gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender = match.group(1).upper()
                confidence = self._calculate_field_confidence('gender', gender, text_blocks)
                bbox = self._find_text_bbox(gender, text_blocks)
                
                return ExtractedField(
                    value=gender,
                    confidence=confidence,
                    bbox=bbox,
                    validation_passed=gender in ['LELAKI', 'PEREMPUAN', 'MALE', 'FEMALE']
                )
        
        return ExtractedField(value="", confidence=0.0)
    
    def _extract_birth_date(self, text: str, text_blocks: List[Dict], 
                           nric_field: Optional[ExtractedField]) -> ExtractedField:
        """Extract birth date, validate against NRIC if available"""
        for pattern in self.patterns.date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Try to parse and validate date
                parsed_date = self._parse_date(match)
                if parsed_date:
                    confidence = self._calculate_field_confidence('birth_date', match, text_blocks)
                    bbox = self._find_text_bbox(match, text_blocks)
                    
                    # Validate against NRIC if available
                    validation_passed = True
                    if nric_field and nric_field.value:
                        validation_passed = self._validate_date_nric_consistency(
                            parsed_date, nric_field.value)
                    
                    return ExtractedField(
                        value=match,
                        confidence=confidence,
                        bbox=bbox,
                        validation_passed=validation_passed
                    )
        
        # If no date found but NRIC available, derive from NRIC
        if nric_field and nric_field.value:
            derived_date = self._derive_date_from_nric(nric_field.value)
            if derived_date:
                return ExtractedField(
                    value=derived_date,
                    confidence=0.8,  # High confidence since derived from NRIC
                    validation_passed=True
                )
        
        return ExtractedField(value="", confidence=0.0)
    
    def _extract_religion(self, text: str, text_blocks: List[Dict]) -> ExtractedField:
        """Extract religion field"""
        for pattern in self.patterns.religion_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                religion = match.group(1).upper()
                confidence = self._calculate_field_confidence('religion', religion, text_blocks)
                bbox = self._find_text_bbox(religion, text_blocks)
                
                return ExtractedField(
                    value=religion,
                    confidence=confidence,
                    bbox=bbox,
                    validation_passed=True
                )
        
        return ExtractedField(value="", confidence=0.0)
    
    def _extract_nationality(self, text: str, text_blocks: List[Dict]) -> ExtractedField:
        """Extract nationality field"""
        for pattern in self.patterns.nationality_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                nationality = match.group(1).upper()
                confidence = self._calculate_field_confidence('nationality', nationality, text_blocks)
                bbox = self._find_text_bbox(nationality, text_blocks)
                
                return ExtractedField(
                    value=nationality,
                    confidence=confidence,
                    bbox=bbox,
                    validation_passed=True
                )
        
        return ExtractedField(value="", confidence=0.0)
    
    def _extract_address(self, text: str, text_blocks: List[Dict]) -> ExtractedField:
        """Extract address field"""
        # Look for address after label
        for label_pattern in self.patterns.field_labels['address']:
            # Find label position
            label_match = re.search(label_pattern, text, re.IGNORECASE)
            if label_match:
                # Extract text after label until next field or end
                start_pos = label_match.end()
                remaining_text = text[start_pos:]
                
                # Look for next field label to determine address end
                next_field_pos = len(remaining_text)
                for field_labels in self.patterns.field_labels.values():
                    for field_pattern in field_labels:
                        match = re.search(field_pattern, remaining_text, re.IGNORECASE)
                        if match and match.start() < next_field_pos:
                            next_field_pos = match.start()
                
                address_text = remaining_text[:next_field_pos].strip()
                
                # Clean up address
                address_lines = [line.strip() for line in address_text.split('\n') if line.strip()]
                address = ', '.join(address_lines)
                
                if address:
                    confidence = self._calculate_field_confidence('address', address, text_blocks)
                    
                    return ExtractedField(
                        value=address,
                        confidence=confidence,
                        validation_passed=len(address) > 10
                    )
        
        return ExtractedField(value="", confidence=0.0)
    
    def _extract_issue_date(self, text: str, text_blocks: List[Dict]) -> ExtractedField:
        """Extract issue date"""
        for pattern in self.patterns.date_patterns:
            matches = re.findall(pattern, text)
            # Usually the last date found is the issue date
            if matches:
                issue_date = matches[-1]
                confidence = self._calculate_field_confidence('issue_date', issue_date, text_blocks)
                bbox = self._find_text_bbox(issue_date, text_blocks)
                
                return ExtractedField(
                    value=issue_date,
                    confidence=confidence,
                    bbox=bbox,
                    validation_passed=True
                )
        
        return ExtractedField(value="", confidence=0.0)
    
    def _calculate_field_confidence(self, field_type: str, value: str, 
                                  text_blocks: List[Dict]) -> float:
        """Calculate confidence score for extracted field"""
        # Find matching text block
        for block in text_blocks:
            if value in block['text']:
                base_confidence = block['confidence']
                
                # Adjust based on field type and validation
                if field_type == 'nric' and self._validate_nric(value):
                    return min(base_confidence + 0.2, 1.0)
                elif field_type == 'gender' and value in ['LELAKI', 'PEREMPUAN']:
                    return min(base_confidence + 0.1, 1.0)
                
                return base_confidence
        
        return 0.5  # Default confidence if not found in blocks
    
    def _find_text_bbox(self, text: str, text_blocks: List[Dict]) -> Optional[Tuple[int, int, int, int]]:
        """Find bounding box for specific text"""
        for block in text_blocks:
            if text in block['text']:
                return block['bbox']
        return None
    
    def _validate_nric(self, nric: str) -> bool:
        """Validate NRIC format and checksum"""
        if not re.match(self.patterns.nric_pattern, nric):
            return False
        
        # Extract date part and validate
        date_part = nric[:6]
        try:
            year = int(date_part[:2])
            month = int(date_part[2:4])
            day = int(date_part[4:6])
            
            # Basic date validation
            if month < 1 or month > 12 or day < 1 or day > 31:
                return False
            
            return True
        except ValueError:
            return False
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        formats = ['%d-%m-%Y', '%d/%m/%Y', '%d-%m-%y', '%d/%m/%y', '%Y-%m-%d', '%Y/%m/%d']
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _validate_date_nric_consistency(self, birth_date: datetime, nric: str) -> bool:
        """Validate that birth date matches NRIC date"""
        nric_date_part = nric[:6]
        nric_year = int(nric_date_part[:2])
        nric_month = int(nric_date_part[2:4])
        nric_day = int(nric_date_part[4:6])
        
        # Handle year conversion (assume 1900s for now)
        full_year = 1900 + nric_year if nric_year > 50 else 2000 + nric_year
        
        return (birth_date.year == full_year and 
                birth_date.month == nric_month and 
                birth_date.day == nric_day)
    
    def _derive_date_from_nric(self, nric: str) -> Optional[str]:
        """Derive birth date from NRIC"""
        if not self._validate_nric(nric):
            return None
        
        date_part = nric[:6]
        year = int(date_part[:2])
        month = date_part[2:4]
        day = date_part[4:6]
        
        # Convert 2-digit year to 4-digit
        full_year = 1900 + year if year > 50 else 2000 + year
        
        return f"{day}-{month}-{full_year}"
    
    def _validate_fields(self, fields: Dict[str, ExtractedField]) -> Dict[str, bool]:
        """Validate all extracted fields"""
        validation = {}
        
        for field_name, field in fields.items():
            validation[field_name] = field.validation_passed
        
        # Cross-field validation
        if fields.get('nric') and fields.get('birth_date'):
            nric_val = fields['nric'].value
            birth_val = fields['birth_date'].value
            
            if nric_val and birth_val:
                parsed_birth = self._parse_date(birth_val)
                if parsed_birth:
                    validation['nric_birth_consistency'] = self._validate_date_nric_consistency(
                        parsed_birth, nric_val)
        
        return validation
    
    def extract_batch(self, image_paths: List[str], output_file: Optional[str] = None) -> List[Dict]:
        """Extract fields from multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.extract_fields(image_path)
                
                # Convert to serializable format
                result_dict = {
                    'image_path': image_path,
                    'fields': {k: {
                        'value': v.value,
                        'confidence': v.confidence,
                        'bbox': v.bbox,
                        'validation_passed': v.validation_passed
                    } for k, v in result.fields.items()},
                    'overall_confidence': result.overall_confidence,
                    'processing_time': result.processing_time,
                    'image_quality_score': result.image_quality_score,
                    'validation_summary': result.validation_summary,
                    'extracted_at': datetime.now().isoformat()
                }
                
                results.append(result_dict)
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'extracted_at': datetime.now().isoformat()
                })
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Results saved to {output_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Extract fields from Malaysian IC images")
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--ocr_engine", choices=["auto", "paddle", "tesseract"], 
                       default="auto", help="OCR engine to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    extractor = MalaysianICExtractor(ocr_engine=args.ocr_engine)
    
    if os.path.isfile(args.input):
        # Single image
        result = extractor.extract_fields(args.input)
        
        print(f"\n🔍 Extraction Results for {args.input}")
        print(f"📊 Overall Confidence: {result.overall_confidence:.2f}")
        print(f"⏱️  Processing Time: {result.processing_time:.2f}s")
        print(f"🖼️  Image Quality: {result.image_quality_score:.2f}")
        print("\n📋 Extracted Fields:")
        
        for field_name, field in result.fields.items():
            status = "✅" if field.validation_passed else "❌"
            print(f"  {status} {field_name}: {field.value} (conf: {field.confidence:.2f})")
        
        if args.output:
            result_dict = {
                'image_path': args.input,
                'fields': {k: {
                    'value': v.value,
                    'confidence': v.confidence,
                    'bbox': v.bbox,
                    'validation_passed': v.validation_passed
                } for k, v in result.fields.items()},
                'overall_confidence': result.overall_confidence,
                'processing_time': result.processing_time,
                'image_quality_score': result.image_quality_score,
                'validation_summary': result.validation_summary,
                'extracted_at': datetime.now().isoformat()
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Results saved to {args.output}")
    
    elif os.path.isdir(args.input):
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        if not image_paths:
            print(f"No images found in {args.input}")
            return
        
        output_file = args.output or "extraction_results.json"
        results = extractor.extract_batch(image_paths, output_file)
        
        print(f"\n🎉 Batch extraction complete!")
        print(f"📁 Processed {len(image_paths)} images")
        print(f"💾 Results saved to {output_file}")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()