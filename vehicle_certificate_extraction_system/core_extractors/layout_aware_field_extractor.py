#!/usr/bin/env python3
"""
Layout-Aware Field Extractor for Malaysian Vehicle Registration Certificates

This module provides advanced field extraction using spatial analysis, context understanding,
and template matching to accurately identify and extract document fields.
"""

import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
import re
from pathlib import Path
import argparse
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates and confidence."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height

@dataclass
class TextRegion:
    """Represents a detected text region with OCR results."""
    bbox: BoundingBox
    text: str
    confidence: float
    level: int
    field_type: Optional[str] = None
    is_label: bool = False
    is_value: bool = False

@dataclass
class ExtractedField:
    """Represents an extracted field with label and value."""
    field_name: str
    label_region: Optional[TextRegion]
    value_region: TextRegion
    confidence: float
    extraction_method: str
    spatial_context: Dict[str, Any]

class SpatialAnalyzer:
    """Analyzes spatial relationships between text regions."""
    
    def __init__(self):
        self.proximity_threshold = 50  # pixels
        self.alignment_threshold = 10  # pixels
    
    def find_nearby_regions(self, target_region: TextRegion, 
                          all_regions: List[TextRegion], 
                          direction: str = "right") -> List[TextRegion]:
        """Find text regions in a specific direction from the target region."""
        nearby = []
        target_center = target_region.bbox.center
        
        for region in all_regions:
            if region == target_region:
                continue
            
            region_center = region.bbox.center
            distance = self._calculate_distance(target_center, region_center)
            
            if distance > self.proximity_threshold:
                continue
            
            # Check direction
            if direction == "right":
                if (region_center[0] > target_center[0] and 
                    abs(region_center[1] - target_center[1]) < self.alignment_threshold):
                    nearby.append(region)
            elif direction == "below":
                if (region_center[1] > target_center[1] and 
                    abs(region_center[0] - target_center[0]) < self.alignment_threshold):
                    nearby.append(region)
            elif direction == "left":
                if (region_center[0] < target_center[0] and 
                    abs(region_center[1] - target_center[1]) < self.alignment_threshold):
                    nearby.append(region)
            elif direction == "above":
                if (region_center[1] < target_center[1] and 
                    abs(region_center[0] - target_center[0]) < self.alignment_threshold):
                    nearby.append(region)
        
        # Sort by distance
        nearby.sort(key=lambda r: self._calculate_distance(target_center, r.bbox.center))
        return nearby
    
    def find_regions_in_area(self, bbox: BoundingBox, 
                           all_regions: List[TextRegion]) -> List[TextRegion]:
        """Find all text regions within a specific bounding box area."""
        regions_in_area = []
        
        for region in all_regions:
            if self._bbox_overlap(bbox, region.bbox) > 0.1:  # 10% overlap threshold
                regions_in_area.append(region)
        
        return regions_in_area
    
    def group_regions_by_rows(self, regions: List[TextRegion]) -> List[List[TextRegion]]:
        """Group text regions into rows based on vertical alignment."""
        if not regions:
            return []
        
        # Sort by y-coordinate
        sorted_regions = sorted(regions, key=lambda r: r.bbox.y)
        
        rows = []
        current_row = [sorted_regions[0]]
        current_y = sorted_regions[0].bbox.y
        
        for region in sorted_regions[1:]:
            if abs(region.bbox.y - current_y) < self.alignment_threshold:
                current_row.append(region)
            else:
                # Sort current row by x-coordinate
                current_row.sort(key=lambda r: r.bbox.x)
                rows.append(current_row)
                current_row = [region]
                current_y = region.bbox.y
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda r: r.bbox.x)
            rows.append(current_row)
        
        return rows
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _bbox_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        # Calculate intersection
        x1 = max(bbox1.x, bbox2.x)
        y1 = max(bbox1.y, bbox2.y)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = bbox1.area + bbox2.area - intersection
        
        return intersection / union if union > 0 else 0.0

class ContextAnalyzer:
    """Analyzes text context to identify labels and values."""
    
    def __init__(self):
        # Malaysian field labels and their variations
        self.field_labels = {
            'plate_number': [
                'no pendaftaran', 'no. pendaftaran', 'pendaftaran', 'registration no',
                'no pendaftaran kenderaan', 'plate number'
            ],
            'owner_name': [
                'nama pemilik', 'nama', 'name', 'owner name', 'pemilik'
            ],
            'nric': [
                'no kad pengenalan', 'no. kad pengenalan', 'kad pengenalan', 'i.c. no',
                'ic no', 'nric', 'identity card'
            ],
            'address': [
                'alamat', 'address'
            ],
            'make': [
                'jenama', 'make', 'brand'
            ],
            'model': [
                'model'
            ],
            'year': [
                'tahun dibuat', 'tahun', 'year made', 'year', 'dibuat'
            ],
            'engine_number': [
                'no enjin', 'no. enjin', 'enjin', 'engine no', 'engine number'
            ],
            'chassis_number': [
                'no casis', 'no. casis', 'casis', 'chassis no', 'chassis number'
            ],
            'color': [
                'warna', 'color', 'colour'
            ],
            'fuel_type': [
                'jenis bahan api', 'bahan api', 'fuel type', 'fuel'
            ],
            'engine_capacity': [
                'isi padu', 'engine capacity', 'capacity', 'cc'
            ]
        }
        
        # Value patterns for validation
        self.value_patterns = {
            'plate_number': r'^[A-Z]{1,3}\s*\d{1,4}\s*[A-Z]?$',
            'nric': r'^\d{6}-\d{2}-\d{4}$',
            'year': r'^(19|20)\d{2}$',
            'engine_capacity': r'^\d+(\.\d+)?\s*(cc|CC|l|L)?$'
        }
    
    def identify_labels_and_values(self, regions: List[TextRegion]) -> Tuple[List[TextRegion], List[TextRegion]]:
        """Identify which regions are labels and which are values."""
        labels = []
        values = []
        
        for region in regions:
            text = region.text.lower().strip()
            
            # Check if this looks like a label
            is_label = self._is_likely_label(text)
            
            if is_label:
                region.is_label = True
                labels.append(region)
            else:
                region.is_value = True
                values.append(region)
        
        return labels, values
    
    def match_label_to_field(self, label_text: str) -> Optional[str]:
        """Match a label text to a known field type."""
        label_text = label_text.lower().strip()
        
        for field_name, label_variations in self.field_labels.items():
            for variation in label_variations:
                if variation in label_text or label_text in variation:
                    return field_name
        
        return None
    
    def validate_value_for_field(self, value_text: str, field_name: str) -> bool:
        """Validate if a value matches the expected pattern for a field."""
        if field_name not in self.value_patterns:
            return True  # No specific pattern to validate
        
        pattern = self.value_patterns[field_name]
        return bool(re.match(pattern, value_text.strip()))
    
    def _is_likely_label(self, text: str) -> bool:
        """Determine if text is likely a field label."""
        # Check for common label indicators
        label_indicators = [
            'no.', 'no', 'nama', 'alamat', 'tahun', 'jenama', 'model',
            'warna', 'enjin', 'casis', 'kad', 'pengenalan', 'pemilik',
            'dibuat', 'jenis', 'bahan', 'api', 'isi', 'padu'
        ]
        
        text_lower = text.lower()
        
        # Check if contains label indicators
        for indicator in label_indicators:
            if indicator in text_lower:
                return True
        
        # Check if ends with colon
        if text.endswith(':'):
            return True
        
        # Check if it's all uppercase (common for labels)
        if text.isupper() and len(text) > 2:
            return True
        
        return False

class TemplateBasedExtractor:
    """Extracts fields using template matching and spatial relationships."""
    
    def __init__(self, template_path: Optional[str] = None):
        self.template = None
        self.spatial_analyzer = SpatialAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        
        if template_path:
            self.load_template(template_path)
    
    def load_template(self, template_path: str):
        """Load document template for field matching."""
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = json.load(f)
        logger.info(f"Loaded template with {len(self.template.get('fields', []))} fields")
    
    def extract_fields(self, ocr_results: Dict[str, Any]) -> List[ExtractedField]:
        """Extract fields from OCR results using multiple strategies."""
        # Convert OCR results to TextRegion objects
        text_regions = self._convert_ocr_to_regions(ocr_results)
        
        # Identify labels and values
        labels, values = self.context_analyzer.identify_labels_and_values(text_regions)
        
        logger.info(f"Identified {len(labels)} labels and {len(values)} values")
        
        # Extract fields using different strategies
        extracted_fields = []
        
        # Strategy 1: Label-value pairs
        extracted_fields.extend(self._extract_label_value_pairs(labels, values, text_regions))
        
        # Strategy 2: Template-based extraction
        if self.template:
            extracted_fields.extend(self._extract_using_template(text_regions))
        
        # Strategy 3: Pattern-based extraction
        extracted_fields.extend(self._extract_using_patterns(text_regions))
        
        # Remove duplicates and merge results
        final_fields = self._merge_and_deduplicate(extracted_fields)
        
        return final_fields
    
    def _convert_ocr_to_regions(self, ocr_results: Dict[str, Any]) -> List[TextRegion]:
        """Convert OCR results to TextRegion objects."""
        regions = []
        
        for region_data in ocr_results.get('all_text_regions', []):
            bbox_data = region_data['bbox']
            bbox = BoundingBox(
                x=bbox_data['x'],
                y=bbox_data['y'],
                width=bbox_data['width'],
                height=bbox_data['height'],
                confidence=bbox_data['confidence']
            )
            
            region = TextRegion(
                bbox=bbox,
                text=region_data['text'],
                confidence=region_data['confidence'],
                level=region_data['level']
            )
            regions.append(region)
        
        return regions
    
    def _extract_label_value_pairs(self, labels: List[TextRegion], 
                                 values: List[TextRegion], 
                                 all_regions: List[TextRegion]) -> List[ExtractedField]:
        """Extract fields by finding label-value pairs."""
        extracted = []
        used_values = set()
        
        for label in labels:
            # Try to match label to a known field
            field_name = self.context_analyzer.match_label_to_field(label.text)
            if not field_name:
                continue
            
            # Find nearby values
            nearby_right = self.spatial_analyzer.find_nearby_regions(label, values, "right")
            nearby_below = self.spatial_analyzer.find_nearby_regions(label, values, "below")
            
            # Prefer values to the right, then below
            candidate_values = nearby_right + nearby_below
            
            for value in candidate_values:
                if id(value) in used_values:
                    continue
                
                # Validate value if pattern exists
                if self.context_analyzer.validate_value_for_field(value.text, field_name):
                    extracted_field = ExtractedField(
                        field_name=field_name,
                        label_region=label,
                        value_region=value,
                        confidence=(label.confidence + value.confidence) / 2,
                        extraction_method="label_value_pair",
                        spatial_context={
                            "label_position": asdict(label.bbox),
                            "value_position": asdict(value.bbox),
                            "distance": self.spatial_analyzer._calculate_distance(
                                label.bbox.center, value.bbox.center
                            )
                        }
                    )
                    extracted.append(extracted_field)
                    used_values.add(id(value))
                    break
        
        return extracted
    
    def _extract_using_template(self, regions: List[TextRegion]) -> List[ExtractedField]:
        """Extract fields using template-based spatial matching."""
        extracted = []
        
        if not self.template or 'fields' not in self.template:
            return extracted
        
        template_fields = self.template['fields']
        
        for template_field in template_fields:
            # Handle both list and dict formats for template fields
            if isinstance(template_field, dict):
                field_name = template_field.get('label', '').lower().replace(' ', '_')
                template_bbox_raw = template_field.get('bbox', [])
            else:
                # Skip if not a dict (might be a list item we can't process)
                continue
            
            if not template_bbox_raw or not field_name:
                continue
            
            # Convert bbox from list [x, y, width, height] to dict format
            if isinstance(template_bbox_raw, list) and len(template_bbox_raw) >= 4:
                template_bbox = {
                    'x': template_bbox_raw[0],
                    'y': template_bbox_raw[1], 
                    'width': template_bbox_raw[2],
                    'height': template_bbox_raw[3]
                }
            elif isinstance(template_bbox_raw, dict):
                template_bbox = template_bbox_raw
            else:
                continue
            
            # Create search area around template position (with some tolerance)
            search_bbox = BoundingBox(
                x=max(0, template_bbox.get('x', 0) - 20),
                y=max(0, template_bbox.get('y', 0) - 20),
                width=template_bbox.get('width', 50) + 40,
                height=template_bbox.get('height', 20) + 40
            )
            
            # Find regions in the search area
            nearby_regions = self.spatial_analyzer.find_regions_in_area(search_bbox, regions)
            
            if nearby_regions:
                # Choose the region with highest confidence
                best_region = max(nearby_regions, key=lambda r: r.confidence)
                
                extracted_field = ExtractedField(
                    field_name=field_name,
                    label_region=None,
                    value_region=best_region,
                    confidence=best_region.confidence * 0.8,  # Slightly lower confidence for template matching
                    extraction_method="template_based",
                    spatial_context={
                        "template_position": template_bbox,
                        "actual_position": asdict(best_region.bbox),
                        "search_area": asdict(search_bbox)
                    }
                )
                extracted.append(extracted_field)
        
        return extracted
    
    def _extract_using_patterns(self, regions: List[TextRegion]) -> List[ExtractedField]:
        """Extract fields using pattern matching without labels."""
        extracted = []
        
        for region in regions:
            text = region.text.strip()
            
            # Check against known patterns
            for field_name, pattern in self.context_analyzer.value_patterns.items():
                if re.match(pattern, text):
                    extracted_field = ExtractedField(
                        field_name=field_name,
                        label_region=None,
                        value_region=region,
                        confidence=region.confidence * 0.7,  # Lower confidence for pattern-only matching
                        extraction_method="pattern_based",
                        spatial_context={
                            "pattern": pattern,
                            "position": asdict(region.bbox)
                        }
                    )
                    extracted.append(extracted_field)
        
        return extracted
    
    def _merge_and_deduplicate(self, extracted_fields: List[ExtractedField]) -> List[ExtractedField]:
        """Merge and deduplicate extracted fields, keeping the best ones."""
        field_groups = defaultdict(list)
        
        # Group by field name
        for field in extracted_fields:
            field_groups[field.field_name].append(field)
        
        final_fields = []
        
        for field_name, candidates in field_groups.items():
            if not candidates:
                continue
            
            # Sort by confidence and extraction method preference
            method_priority = {"label_value_pair": 3, "template_based": 2, "pattern_based": 1}
            
            candidates.sort(key=lambda f: (
                method_priority.get(f.extraction_method, 0),
                f.confidence
            ), reverse=True)
            
            # Take the best candidate
            best_field = candidates[0]
            final_fields.append(best_field)
            
            logger.info(f"Selected field '{field_name}': '{best_field.value_region.text}' "
                       f"(method: {best_field.extraction_method}, confidence: {best_field.confidence:.1f})")
        
        return final_fields

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Layout-Aware Field Extractor for Malaysian Vehicle Registration Certificates")
    parser.add_argument("ocr_results", help="Path to OCR results JSON file")
    parser.add_argument("--template", help="Path to document template JSON file")
    parser.add_argument("--output", help="Output file for extracted fields")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load OCR results
    with open(args.ocr_results, 'r', encoding='utf-8') as f:
        ocr_results = json.load(f)
    
    # Initialize extractor
    extractor = TemplateBasedExtractor(template_path=args.template)
    
    # Extract fields
    try:
        extracted_fields = extractor.extract_fields(ocr_results)
        
        # Prepare output
        output_data = {
            "source_file": args.ocr_results,
            "extraction_timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "total_fields_extracted": len(extracted_fields),
            "extracted_fields": {
                field.field_name: {
                    "text": field.value_region.text,
                    "confidence": field.confidence,
                    "extraction_method": field.extraction_method,
                    "bbox": asdict(field.value_region.bbox),
                    "spatial_context": field.spatial_context,
                    "label_text": field.label_region.text if field.label_region else None
                }
                for field in extracted_fields
            }
        }
        
        # Save results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Extracted fields saved to {args.output}")
        
        # Print results
        print(f"\nLayout-Aware Field Extraction Results")
        print("=" * 50)
        print(f"Total fields extracted: {len(extracted_fields)}")
        
        for field in extracted_fields:
            print(f"\n{field.field_name.upper()}:")
            print(f"  Value: {field.value_region.text}")
            print(f"  Confidence: {field.confidence:.1f}")
            print(f"  Method: {field.extraction_method}")
            if field.label_region:
                print(f"  Label: {field.label_region.text}")
        
    except Exception as e:
        logger.error(f"Error extracting fields: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())