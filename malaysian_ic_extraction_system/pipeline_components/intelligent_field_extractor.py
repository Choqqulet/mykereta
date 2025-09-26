#!/usr/bin/env python3
"""
Intelligent Field Extraction System

Advanced field extraction system for Malaysian IC and Passport documents using
pattern matching, NLP, and machine learning techniques.
"""

import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
import numpy as np
from collections import defaultdict
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FieldMatch:
    """Individual field match result"""
    field_name: str
    value: str
    confidence: float
    method: str  # pattern, nlp, ml, coordinate
    bbox: Optional[Tuple[int, int, int, int]] = None
    context: str = ""
    alternatives: List[str] = field(default_factory=list)

@dataclass
class ExtractionResult:
    """Complete field extraction result"""
    extracted_fields: Dict[str, FieldMatch] = field(default_factory=dict)
    document_type: str = "unknown"
    confidence_score: float = 0.0
    processing_time: float = 0.0
    extraction_methods_used: List[str] = field(default_factory=list)
    raw_text: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class IntelligentFieldExtractor:
    """
    Intelligent field extraction system for IC and Passport documents.
    
    Features:
    - Multi-method extraction (regex patterns, NLP, ML, coordinate-based)
    - Document type detection and specialized extraction
    - Confidence scoring and alternative suggestions
    - Context-aware field validation
    - Spatial relationship analysis for coordinate-based extraction
    """
    
    def __init__(self, 
                 spacy_model: str = "en_core_web_sm",
                 enable_ml_extraction: bool = True,
                 confidence_threshold: float = 0.6):
        """
        Initialize the intelligent field extractor.
        
        Args:
            spacy_model: SpaCy model for NLP processing
            enable_ml_extraction: Enable ML-based extraction methods
            confidence_threshold: Minimum confidence for field acceptance
        """
        self.confidence_threshold = confidence_threshold
        self.enable_ml_extraction = enable_ml_extraction
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load(spacy_model)
            self.matcher = Matcher(self.nlp.vocab)
            self._initialize_patterns()
            logger.info(f"NLP components initialized with model: {spacy_model}")
        except OSError:
            logger.warning(f"SpaCy model {spacy_model} not found. NLP extraction disabled.")
            self.nlp = None
            self.matcher = None
        
        # Initialize extraction patterns
        self.patterns = self._load_extraction_patterns()
        self.field_keywords = self._load_field_keywords()
        self.document_layouts = self._load_document_layouts()
        
        logger.info("Intelligent Field Extractor initialized")
    
    def extract_fields(self, 
                      ocr_result: Dict[str, Any],
                      document_type: str = "auto",
                      use_coordinates: bool = True) -> ExtractionResult:
        """
        Extract fields from OCR result using multiple methods.
        
        Args:
            ocr_result: OCR result with text and optional coordinate information
            document_type: Document type hint ('ic', 'passport', 'auto')
            use_coordinates: Use coordinate information for spatial extraction
            
        Returns:
            ExtractionResult: Complete extraction results
        """
        start_time = datetime.now()
        
        try:
            # Extract text and coordinates
            text = ocr_result.get('text', '')
            word_boxes = ocr_result.get('word_boxes', [])
            
            if not text.strip():
                return ExtractionResult(
                    raw_text=text,
                    errors=["No text found in OCR result"]
                )
            
            # Step 1: Document type detection
            if document_type == "auto":
                document_type = self._detect_document_type(text, word_boxes)
            
            # Step 2: Multi-method field extraction
            extraction_methods = []
            all_matches = {}
            
            # Pattern-based extraction
            pattern_matches = self._extract_with_patterns(text, document_type)
            all_matches.update(pattern_matches)
            if pattern_matches:
                extraction_methods.append("pattern_matching")
            
            # NLP-based extraction
            if self.nlp:
                nlp_matches = self._extract_with_nlp(text, document_type)
                all_matches = self._merge_matches(all_matches, nlp_matches)
                if nlp_matches:
                    extraction_methods.append("nlp_extraction")
            
            # Coordinate-based extraction
            if use_coordinates and word_boxes:
                coord_matches = self._extract_with_coordinates(text, word_boxes, document_type)
                all_matches = self._merge_matches(all_matches, coord_matches)
                if coord_matches:
                    extraction_methods.append("coordinate_based")
            
            # ML-based extraction (if enabled)
            if self.enable_ml_extraction:
                ml_matches = self._extract_with_ml(text, document_type)
                all_matches = self._merge_matches(all_matches, ml_matches)
                if ml_matches:
                    extraction_methods.append("ml_extraction")
            
            # Step 3: Post-processing and validation
            final_fields = self._post_process_matches(all_matches, document_type)
            
            # Step 4: Calculate confidence and create structured data
            confidence_score = self._calculate_extraction_confidence(final_fields)
            structured_data = self._create_structured_data(final_fields, document_type)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ExtractionResult(
                extracted_fields=final_fields,
                document_type=document_type,
                confidence_score=confidence_score,
                processing_time=processing_time,
                extraction_methods_used=extraction_methods,
                raw_text=text,
                structured_data=structured_data
            )
            
        except Exception as e:
            logger.error(f"Field extraction error: {e}")
            return ExtractionResult(
                raw_text=ocr_result.get('text', ''),
                errors=[f"Extraction error: {str(e)}"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _detect_document_type(self, text: str, word_boxes: List[Dict]) -> str:
        """Detect document type from text content"""
        try:
            text_lower = text.lower()
            
            # Malaysian IC indicators
            ic_indicators = [
                'kad pengenalan', 'identity card', 'mykad',
                'warganegara malaysia', 'malaysian citizen'
            ]
            
            # Passport indicators
            passport_indicators = [
                'passport', 'pasport', 'travel document',
                'republic of', 'kingdom of', 'people\'s republic'
            ]
            
            # Check for IC patterns
            if any(indicator in text_lower for indicator in ic_indicators):
                return 'ic'
            
            # Check for IC number pattern
            if re.search(r'\d{6}[-\s]?\d{2}[-\s]?\d{4}', text):
                return 'ic'
            
            # Check for passport patterns
            if any(indicator in text_lower for indicator in passport_indicators):
                return 'passport'
            
            # Check for passport number pattern
            if re.search(r'[A-Z]\d{8}', text):
                return 'passport'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Document type detection error: {e}")
            return 'unknown'
    
    def _extract_with_patterns(self, text: str, document_type: str) -> Dict[str, FieldMatch]:
        """Extract fields using regex patterns"""
        matches = {}
        
        try:
            # Get patterns for document type
            doc_patterns = self.patterns.get(document_type, {})
            general_patterns = self.patterns.get('general', {})
            
            # Combine patterns
            all_patterns = {**general_patterns, **doc_patterns}
            
            for field_name, pattern_info in all_patterns.items():
                pattern = pattern_info['pattern']
                confidence = pattern_info.get('confidence', 0.8)
                
                # Find all matches
                pattern_matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in pattern_matches:
                    value = match.group(1) if match.groups() else match.group(0)
                    value = value.strip()
                    
                    if value and len(value) > 1:  # Basic validation
                        # Calculate context
                        start_pos = max(0, match.start() - 50)
                        end_pos = min(len(text), match.end() + 50)
                        context = text[start_pos:end_pos]
                        
                        field_match = FieldMatch(
                            field_name=field_name,
                            value=value,
                            confidence=confidence,
                            method="pattern",
                            context=context
                        )
                        
                        # Keep the best match for each field
                        if field_name not in matches or field_match.confidence > matches[field_name].confidence:
                            matches[field_name] = field_match
            
            return matches
            
        except Exception as e:
            logger.error(f"Pattern extraction error: {e}")
            return {}
    
    def _extract_with_nlp(self, text: str, document_type: str) -> Dict[str, FieldMatch]:
        """Extract fields using NLP techniques"""
        matches = {}
        
        try:
            if not self.nlp:
                return matches
            
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Named Entity Recognition
            for ent in doc.ents:
                field_name = self._map_entity_to_field(ent.label_, ent.text, document_type)
                if field_name:
                    field_match = FieldMatch(
                        field_name=field_name,
                        value=ent.text,
                        confidence=0.7,  # NLP confidence
                        method="nlp",
                        context=ent.sent.text if ent.sent else ""
                    )
                    
                    if field_name not in matches or field_match.confidence > matches[field_name].confidence:
                        matches[field_name] = field_match
            
            # Pattern matching with spaCy matcher
            matcher_matches = self.matcher(doc)
            spans = [doc[start:end] for match_id, start, end in matcher_matches]
            spans = filter_spans(spans)  # Remove overlapping spans
            
            for span in spans:
                field_name = self._get_pattern_field_name(span.label_)
                if field_name:
                    field_match = FieldMatch(
                        field_name=field_name,
                        value=span.text,
                        confidence=0.8,
                        method="nlp",
                        context=span.sent.text if span.sent else ""
                    )
                    
                    if field_name not in matches or field_match.confidence > matches[field_name].confidence:
                        matches[field_name] = field_match
            
            return matches
            
        except Exception as e:
            logger.error(f"NLP extraction error: {e}")
            return {}
    
    def _extract_with_coordinates(self, 
                                text: str, 
                                word_boxes: List[Dict], 
                                document_type: str) -> Dict[str, FieldMatch]:
        """Extract fields using coordinate information and spatial relationships"""
        matches = {}
        
        try:
            if not word_boxes:
                return matches
            
            # Get document layout information
            layout_info = self.document_layouts.get(document_type, {})
            
            # Create spatial index of words
            spatial_index = self._create_spatial_index(word_boxes)
            
            # Find field labels and their associated values
            for word_info in word_boxes:
                word_text = word_info.get('text', '').strip()
                word_bbox = word_info.get('bbox', [])
                
                if not word_text or not word_bbox:
                    continue
                
                # Check if this word is a field label
                field_name = self._identify_field_label(word_text, document_type)
                if field_name:
                    # Find the associated value using spatial relationships
                    value_candidates = self._find_spatial_value_candidates(
                        word_bbox, 
                        spatial_index, 
                        layout_info.get(field_name, {})
                    )
                    
                    if value_candidates:
                        best_candidate = max(value_candidates, key=lambda x: x['confidence'])
                        
                        field_match = FieldMatch(
                            field_name=field_name,
                            value=best_candidate['text'],
                            confidence=best_candidate['confidence'],
                            method="coordinate",
                            bbox=tuple(best_candidate['bbox']),
                            context=f"Label: {word_text}"
                        )
                        
                        if field_name not in matches or field_match.confidence > matches[field_name].confidence:
                            matches[field_name] = field_match
            
            return matches
            
        except Exception as e:
            logger.error(f"Coordinate extraction error: {e}")
            return {}
    
    def _extract_with_ml(self, text: str, document_type: str) -> Dict[str, FieldMatch]:
        """Extract fields using machine learning models"""
        matches = {}
        
        try:
            # This is a placeholder for ML-based extraction
            # In a real implementation, you would load trained models here
            
            # For now, implement a simple heuristic-based approach
            # that mimics ML behavior
            
            # Tokenize text
            words = text.split()
            
            # Simple sequence labeling simulation
            for i, word in enumerate(words):
                # Look for patterns that might indicate field values
                
                # IC number detection
                if re.match(r'\d{6}[-\s]?\d{2}[-\s]?\d{4}', word):
                    field_match = FieldMatch(
                        field_name="ic_number",
                        value=word,
                        confidence=0.9,
                        method="ml",
                        context=" ".join(words[max(0, i-3):i+4])
                    )
                    matches["ic_number"] = field_match
                
                # Date detection
                elif re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', word):
                    # Determine if it's birth date, issue date, or expiry date based on context
                    context_words = words[max(0, i-5):i+5]
                    context_text = " ".join(context_words).lower()
                    
                    if any(keyword in context_text for keyword in ['birth', 'born', 'lahir']):
                        field_name = "date_of_birth"
                    elif any(keyword in context_text for keyword in ['issue', 'issued', 'dikeluarkan']):
                        field_name = "issue_date"
                    elif any(keyword in context_text for keyword in ['expiry', 'expires', 'tamat']):
                        field_name = "expiry_date"
                    else:
                        field_name = "date"
                    
                    field_match = FieldMatch(
                        field_name=field_name,
                        value=word,
                        confidence=0.8,
                        method="ml",
                        context=" ".join(context_words)
                    )
                    
                    if field_name not in matches or field_match.confidence > matches[field_name].confidence:
                        matches[field_name] = field_match
            
            return matches
            
        except Exception as e:
            logger.error(f"ML extraction error: {e}")
            return {}
    
    def _merge_matches(self, 
                      existing_matches: Dict[str, FieldMatch], 
                      new_matches: Dict[str, FieldMatch]) -> Dict[str, FieldMatch]:
        """Merge field matches from different extraction methods"""
        try:
            merged = existing_matches.copy()
            
            for field_name, new_match in new_matches.items():
                if field_name not in merged:
                    merged[field_name] = new_match
                else:
                    existing_match = merged[field_name]
                    
                    # Choose the match with higher confidence
                    if new_match.confidence > existing_match.confidence:
                        # Keep the old match as an alternative
                        new_match.alternatives.append(existing_match.value)
                        merged[field_name] = new_match
                    else:
                        # Keep the new match as an alternative
                        existing_match.alternatives.append(new_match.value)
            
            return merged
            
        except Exception as e:
            logger.error(f"Match merging error: {e}")
            return existing_matches
    
    def _post_process_matches(self, 
                            matches: Dict[str, FieldMatch], 
                            document_type: str) -> Dict[str, FieldMatch]:
        """Post-process and validate extracted matches"""
        processed_matches = {}
        
        try:
            for field_name, match in matches.items():
                # Apply field-specific cleaning and validation
                cleaned_value = self._clean_field_value(match.value, field_name)
                
                # Validate the cleaned value
                is_valid, validation_confidence = self._validate_extracted_value(
                    cleaned_value, 
                    field_name, 
                    document_type
                )
                
                if is_valid and match.confidence >= self.confidence_threshold:
                    # Adjust confidence based on validation
                    final_confidence = match.confidence * validation_confidence
                    
                    processed_match = FieldMatch(
                        field_name=field_name,
                        value=cleaned_value,
                        confidence=final_confidence,
                        method=match.method,
                        bbox=match.bbox,
                        context=match.context,
                        alternatives=match.alternatives
                    )
                    
                    processed_matches[field_name] = processed_match
            
            return processed_matches
            
        except Exception as e:
            logger.error(f"Post-processing error: {e}")
            return matches
    
    def _calculate_extraction_confidence(self, matches: Dict[str, FieldMatch]) -> float:
        """Calculate overall extraction confidence"""
        try:
            if not matches:
                return 0.0
            
            # Weight different fields by importance
            field_weights = {
                'name': 0.2,
                'ic_number': 0.3,
                'passport_number': 0.3,
                'date_of_birth': 0.2,
                'nationality': 0.15,
                'gender': 0.1,
                'issue_date': 0.1,
                'expiry_date': 0.15
            }
            
            total_weight = 0
            weighted_confidence = 0
            
            for field_name, match in matches.items():
                weight = field_weights.get(field_name, 0.1)
                total_weight += weight
                weighted_confidence += match.confidence * weight
            
            return weighted_confidence / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.0
    
    def _create_structured_data(self, 
                              matches: Dict[str, FieldMatch], 
                              document_type: str) -> Dict[str, Any]:
        """Create structured data from extracted fields"""
        try:
            structured = {
                'document_type': document_type,
                'extraction_timestamp': datetime.now().isoformat(),
                'fields': {}
            }
            
            for field_name, match in matches.items():
                structured['fields'][field_name] = {
                    'value': match.value,
                    'confidence': match.confidence,
                    'method': match.method,
                    'alternatives': match.alternatives
                }
                
                if match.bbox:
                    structured['fields'][field_name]['bbox'] = match.bbox
            
            return structured
            
        except Exception as e:
            logger.error(f"Structured data creation error: {e}")
            return {}
    
    # Helper methods and initialization
    def _initialize_patterns(self):
        """Initialize spaCy patterns for field extraction"""
        if not self.matcher:
            return
        
        try:
            # IC number pattern
            ic_pattern = [
                {"TEXT": {"REGEX": r"\d{6}"}},
                {"TEXT": "-", "OP": "?"},
                {"TEXT": {"REGEX": r"\d{2}"}},
                {"TEXT": "-", "OP": "?"},
                {"TEXT": {"REGEX": r"\d{4}"}}
            ]
            self.matcher.add("IC_NUMBER", [ic_pattern])
            
            # Date patterns
            date_pattern1 = [
                {"TEXT": {"REGEX": r"\d{1,2}"}},
                {"TEXT": {"REGEX": r"[/-]"}},
                {"TEXT": {"REGEX": r"\d{1,2}"}},
                {"TEXT": {"REGEX": r"[/-]"}},
                {"TEXT": {"REGEX": r"\d{4}"}}
            ]
            self.matcher.add("DATE", [date_pattern1])
            
            # Name patterns (after specific keywords)
            name_pattern = [
                {"LOWER": {"IN": ["name", "nama"]}},
                {"TEXT": ":", "OP": "?"},
                {"IS_ALPHA": True, "OP": "+"}
            ]
            self.matcher.add("NAME", [name_pattern])
            
        except Exception as e:
            logger.error(f"Pattern initialization error: {e}")
    
    def _load_extraction_patterns(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load regex patterns for field extraction"""
        return {
            'ic': {
                'ic_number': {
                    'pattern': r'(?:No\.?\s*K\.?P\.?|IC\s*No\.?|No\.\s*Pengenalan)[\s:]*(\d{6}[-\s]?\d{2}[-\s]?\d{4})',
                    'confidence': 0.9
                },
                'name': {
                    'pattern': r'(?:Nama|Name)[\s:]+([A-Z][A-Za-z\s\.\-\']+?)(?:\n|$|(?=\d))',
                    'confidence': 0.8
                },
                'date_of_birth': {
                    'pattern': r'(?:Tarikh\s*Lahir|Date\s*of\s*Birth|D\.O\.B)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                    'confidence': 0.9
                },
                'gender': {
                    'pattern': r'(?:Jantina|Sex|Gender)[\s:]*([A-Za-z]+)',
                    'confidence': 0.8
                },
                'nationality': {
                    'pattern': r'(?:Warganegara|Nationality)[\s:]*([A-Za-z\s]+)',
                    'confidence': 0.8
                }
            },
            'passport': {
                'passport_number': {
                    'pattern': r'(?:Passport\s*No\.?|No\.\s*Pasport)[\s:]*([A-Z]\d{8})',
                    'confidence': 0.9
                },
                'name': {
                    'pattern': r'(?:Name|Nama)[\s:]+([A-Z][A-Za-z\s\.\-\']+?)(?:\n|$)',
                    'confidence': 0.8
                },
                'date_of_birth': {
                    'pattern': r'(?:Date\s*of\s*Birth|D\.O\.B|Tarikh\s*Lahir)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                    'confidence': 0.9
                },
                'nationality': {
                    'pattern': r'(?:Nationality|Warganegara)[\s:]*([A-Za-z\s]+)',
                    'confidence': 0.8
                },
                'issue_date': {
                    'pattern': r'(?:Date\s*of\s*Issue|Issued|Dikeluarkan)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                    'confidence': 0.8
                },
                'expiry_date': {
                    'pattern': r'(?:Date\s*of\s*Expiry|Expires|Tamat\s*Tempoh)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                    'confidence': 0.9
                }
            },
            'general': {
                'email': {
                    'pattern': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                    'confidence': 0.9
                },
                'phone': {
                    'pattern': r'(?:Tel|Phone|Telefon)[\s:]*([+]?[\d\s\-\(\)]{7,15})',
                    'confidence': 0.7
                }
            }
        }
    
    def _load_field_keywords(self) -> Dict[str, List[str]]:
        """Load keywords for field identification"""
        return {
            'name': ['name', 'nama', 'full name', 'nama penuh'],
            'ic_number': ['ic', 'k.p', 'kad pengenalan', 'identity card', 'no. k.p', 'ic no'],
            'passport_number': ['passport', 'pasport', 'passport no', 'no. pasport'],
            'date_of_birth': ['date of birth', 'tarikh lahir', 'd.o.b', 'birth date'],
            'gender': ['gender', 'jantina', 'sex'],
            'nationality': ['nationality', 'warganegara'],
            'issue_date': ['issue date', 'date of issue', 'issued', 'dikeluarkan'],
            'expiry_date': ['expiry date', 'date of expiry', 'expires', 'tamat tempoh'],
            'issuing_authority': ['issued by', 'authority', 'pihak berkuasa']
        }
    
    def _load_document_layouts(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load document layout information for coordinate-based extraction"""
        return {
            'ic': {
                'name': {'direction': 'right', 'max_distance': 200},
                'ic_number': {'direction': 'right', 'max_distance': 150},
                'date_of_birth': {'direction': 'right', 'max_distance': 150},
                'gender': {'direction': 'right', 'max_distance': 100}
            },
            'passport': {
                'name': {'direction': 'right', 'max_distance': 200},
                'passport_number': {'direction': 'right', 'max_distance': 150},
                'nationality': {'direction': 'right', 'max_distance': 150},
                'date_of_birth': {'direction': 'right', 'max_distance': 150}
            }
        }
    
    def _map_entity_to_field(self, entity_label: str, entity_text: str, document_type: str) -> Optional[str]:
        """Map spaCy entity to field name"""
        entity_mapping = {
            'PERSON': 'name',
            'DATE': 'date',
            'GPE': 'nationality',  # Geopolitical entity
            'NORP': 'nationality'  # Nationalities or religious groups
        }
        
        field_name = entity_mapping.get(entity_label)
        
        # Refine based on context and document type
        if field_name == 'date':
            # Determine specific date type based on context
            text_lower = entity_text.lower()
            if any(keyword in text_lower for keyword in ['birth', 'born']):
                return 'date_of_birth'
            elif any(keyword in text_lower for keyword in ['issue', 'issued']):
                return 'issue_date'
            elif any(keyword in text_lower for keyword in ['expiry', 'expires']):
                return 'expiry_date'
        
        return field_name
    
    def _get_pattern_field_name(self, pattern_label: str) -> Optional[str]:
        """Get field name from pattern label"""
        pattern_mapping = {
            'IC_NUMBER': 'ic_number',
            'DATE': 'date',
            'NAME': 'name'
        }
        return pattern_mapping.get(pattern_label)
    
    def _create_spatial_index(self, word_boxes: List[Dict]) -> Dict[str, List[Dict]]:
        """Create spatial index for coordinate-based extraction"""
        spatial_index = defaultdict(list)
        
        for word_info in word_boxes:
            text = word_info.get('text', '').strip()
            if text:
                spatial_index[text.lower()].append(word_info)
        
        return dict(spatial_index)
    
    def _identify_field_label(self, word_text: str, document_type: str) -> Optional[str]:
        """Identify if a word is a field label"""
        word_lower = word_text.lower().strip(':')
        
        for field_name, keywords in self.field_keywords.items():
            if any(keyword in word_lower for keyword in keywords):
                return field_name
        
        return None
    
    def _find_spatial_value_candidates(self, 
                                     label_bbox: List[int], 
                                     spatial_index: Dict[str, List[Dict]], 
                                     layout_info: Dict[str, Any]) -> List[Dict]:
        """Find value candidates based on spatial relationships"""
        candidates = []
        
        try:
            direction = layout_info.get('direction', 'right')
            max_distance = layout_info.get('max_distance', 200)
            
            label_x1, label_y1, label_x2, label_y2 = label_bbox
            label_center_y = (label_y1 + label_y2) / 2
            
            # Search for candidates in all words
            for word_list in spatial_index.values():
                for word_info in word_list:
                    word_bbox = word_info.get('bbox', [])
                    if not word_bbox:
                        continue
                    
                    word_x1, word_y1, word_x2, word_y2 = word_bbox
                    word_center_y = (word_y1 + word_y2) / 2
                    
                    # Check spatial relationship
                    if direction == 'right':
                        # Word should be to the right of label
                        if (word_x1 > label_x2 and 
                            abs(word_center_y - label_center_y) < 30 and  # Same line
                            word_x1 - label_x2 < max_distance):
                            
                            distance = word_x1 - label_x2
                            confidence = max(0.1, 1.0 - (distance / max_distance))
                            
                            candidates.append({
                                'text': word_info.get('text', ''),
                                'bbox': word_bbox,
                                'confidence': confidence
                            })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Spatial candidate search error: {e}")
            return []
    
    def _clean_field_value(self, value: str, field_name: str) -> str:
        """Clean extracted field value"""
        try:
            cleaned = value.strip()
            
            if field_name == 'ic_number':
                # Remove extra characters and format properly
                digits = re.sub(r'[^\d]', '', cleaned)
                if len(digits) == 12:
                    cleaned = f"{digits[:6]}-{digits[6:8]}-{digits[8:]}"
            
            elif field_name == 'name':
                # Proper case formatting
                cleaned = ' '.join(word.capitalize() for word in cleaned.split())
            
            elif field_name in ['date_of_birth', 'issue_date', 'expiry_date']:
                # Standardize date format
                cleaned = re.sub(r'[^\d/\-]', '', cleaned)
            
            elif field_name == 'gender':
                # Normalize gender
                cleaned = cleaned.upper()
                if cleaned in ['M', 'MALE', 'LELAKI']:
                    cleaned = 'MALE'
                elif cleaned in ['F', 'FEMALE', 'PEREMPUAN']:
                    cleaned = 'FEMALE'
            
            return cleaned
            
        except Exception:
            return value
    
    def _validate_extracted_value(self, 
                                value: str, 
                                field_name: str, 
                                document_type: str) -> Tuple[bool, float]:
        """Validate extracted field value"""
        try:
            if not value or not value.strip():
                return False, 0.0
            
            # Field-specific validation
            if field_name == 'ic_number':
                if re.match(r'^\d{6}-\d{2}-\d{4}$', value):
                    return True, 1.0
                elif re.match(r'^\d{12}$', value):
                    return True, 0.8
                else:
                    return False, 0.0
            
            elif field_name == 'passport_number':
                if re.match(r'^[A-Z]\d{8}$', value):
                    return True, 1.0
                elif re.match(r'^[A-Z0-9]{6,12}$', value):
                    return True, 0.7
                else:
                    return False, 0.0
            
            elif field_name == 'name':
                if re.match(r'^[A-Za-z\s\.\-\']{2,}$', value) and len(value.split()) <= 5:
                    return True, 0.9
                else:
                    return False, 0.0
            
            elif field_name in ['date_of_birth', 'issue_date', 'expiry_date']:
                if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}$', value):
                    return True, 0.9
                else:
                    return False, 0.0
            
            elif field_name == 'gender':
                if value.upper() in ['MALE', 'FEMALE', 'M', 'F']:
                    return True, 1.0
                else:
                    return False, 0.0
            
            # Default validation for other fields
            return True, 0.7
            
        except Exception:
            return False, 0.0