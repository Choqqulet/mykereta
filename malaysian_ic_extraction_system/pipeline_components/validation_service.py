#!/usr/bin/env python3
"""
Document Validation Service Module

Comprehensive validation system for extracted IC and Passport data with accuracy
verification, format validation, and cross-field consistency checks.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import hashlib
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Supported document types"""
    MALAYSIAN_IC = "malaysian_ic"
    MALAYSIAN_PASSPORT = "malaysian_passport"
    FOREIGN_PASSPORT = "foreign_passport"
    UNKNOWN = "unknown"

class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # All validations must pass
    MODERATE = "moderate"  # Critical validations must pass
    LENIENT = "lenient"    # Basic format checks only

class FieldType(Enum):
    """Field types for validation"""
    NAME = "name"
    IC_NUMBER = "ic_number"
    PASSPORT_NUMBER = "passport_number"
    DATE_OF_BIRTH = "date_of_birth"
    NATIONALITY = "nationality"
    GENDER = "gender"
    DOCUMENT_TYPE = "document_type"
    ISSUE_DATE = "issue_date"
    EXPIRY_DATE = "expiry_date"
    ISSUING_AUTHORITY = "issuing_authority"
    ADDRESS = "address"
    PLACE_OF_BIRTH = "place_of_birth"

@dataclass
class ValidationRule:
    """Individual validation rule"""
    field_type: FieldType
    rule_name: str
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    required: bool = True
    custom_validator: Optional[callable] = None
    error_message: str = ""
    severity: str = "error"  # error, warning, info

@dataclass
class ValidationResult:
    """Result of field validation"""
    field_name: str
    field_type: FieldType
    original_value: str
    cleaned_value: str
    is_valid: bool
    confidence: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    applied_corrections: List[str] = field(default_factory=list)

@dataclass
class DocumentValidationResult:
    """Complete document validation result"""
    document_type: DocumentType
    validation_level: ValidationLevel
    overall_valid: bool
    confidence_score: float
    field_results: Dict[str, ValidationResult] = field(default_factory=dict)
    cross_field_errors: List[str] = field(default_factory=list)
    security_checks: Dict[str, bool] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    timestamp: str = ""

class DocumentValidationService:
    """
    Comprehensive document validation service for IC and Passport documents.
    
    Features:
    - Format validation for Malaysian IC and Passport numbers
    - Date consistency checks and validation
    - Name format and character validation
    - Cross-field consistency verification
    - Security and authenticity checks
    - Automatic data cleaning and correction
    - Confidence scoring for extracted data
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize the validation service.
        
        Args:
            validation_level: Strictness level for validation
        """
        self.validation_level = validation_level
        self.validation_rules = self._initialize_validation_rules()
        self.malaysian_states = self._load_malaysian_states()
        self.country_codes = self._load_country_codes()
        
        logger.info(f"Document Validation Service initialized with level: {validation_level.value}")
    
    def validate_document(self, 
                         extracted_data: Dict[str, Any],
                         document_type: DocumentType = DocumentType.UNKNOWN,
                         confidence_threshold: float = 0.7) -> DocumentValidationResult:
        """
        Validate extracted document data comprehensively.
        
        Args:
            extracted_data: Dictionary of extracted field values
            document_type: Type of document being validated
            confidence_threshold: Minimum confidence for acceptance
            
        Returns:
            DocumentValidationResult: Complete validation results
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Auto-detect document type if unknown
            if document_type == DocumentType.UNKNOWN:
                document_type = self._detect_document_type(extracted_data)
            
            # Step 2: Validate individual fields
            field_results = {}
            for field_name, value in extracted_data.items():
                if value and str(value).strip():
                    field_type = self._map_field_name_to_type(field_name)
                    if field_type:
                        result = self._validate_field(
                            field_name, 
                            str(value).strip(), 
                            field_type, 
                            document_type
                        )
                        field_results[field_name] = result
            
            # Step 3: Cross-field validation
            cross_field_errors = self._validate_cross_field_consistency(
                field_results, 
                document_type
            )
            
            # Step 4: Security and authenticity checks
            security_checks = self._perform_security_checks(
                field_results, 
                document_type
            )
            
            # Step 5: Calculate overall confidence and validity
            overall_valid, confidence_score = self._calculate_overall_validity(
                field_results, 
                cross_field_errors, 
                security_checks,
                confidence_threshold
            )
            
            # Step 6: Generate recommendations
            recommendations = self._generate_recommendations(
                field_results, 
                cross_field_errors, 
                security_checks
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DocumentValidationResult(
                document_type=document_type,
                validation_level=self.validation_level,
                overall_valid=overall_valid,
                confidence_score=confidence_score,
                field_results=field_results,
                cross_field_errors=cross_field_errors,
                security_checks=security_checks,
                recommendations=recommendations,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Document validation error: {e}")
            return DocumentValidationResult(
                document_type=document_type,
                validation_level=self.validation_level,
                overall_valid=False,
                confidence_score=0.0,
                cross_field_errors=[f"Validation error: {str(e)}"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat()
            )
    
    def _initialize_validation_rules(self) -> Dict[FieldType, List[ValidationRule]]:
        """Initialize validation rules for different field types"""
        rules = {
            FieldType.IC_NUMBER: [
                ValidationRule(
                    field_type=FieldType.IC_NUMBER,
                    rule_name="malaysian_ic_format",
                    pattern=r'^\d{6}-\d{2}-\d{4}$',
                    min_length=12,
                    max_length=14,
                    custom_validator=self._validate_malaysian_ic,
                    error_message="Invalid Malaysian IC format (should be YYMMDD-SS-NNNN)"
                )
            ],
            
            FieldType.PASSPORT_NUMBER: [
                ValidationRule(
                    field_type=FieldType.PASSPORT_NUMBER,
                    rule_name="malaysian_passport_format",
                    pattern=r'^[A-Z]\d{8}$',
                    min_length=9,
                    max_length=9,
                    custom_validator=self._validate_malaysian_passport,
                    error_message="Invalid Malaysian passport format (should be A12345678)"
                ),
                ValidationRule(
                    field_type=FieldType.PASSPORT_NUMBER,
                    rule_name="international_passport_format",
                    pattern=r'^[A-Z0-9]{6,12}$',
                    min_length=6,
                    max_length=12,
                    error_message="Invalid international passport format"
                )
            ],
            
            FieldType.NAME: [
                ValidationRule(
                    field_type=FieldType.NAME,
                    rule_name="name_format",
                    pattern=r'^[A-Za-z\s\.\-\']+$',
                    min_length=2,
                    max_length=100,
                    custom_validator=self._validate_name,
                    error_message="Name contains invalid characters"
                )
            ],
            
            FieldType.DATE_OF_BIRTH: [
                ValidationRule(
                    field_type=FieldType.DATE_OF_BIRTH,
                    rule_name="date_format",
                    custom_validator=self._validate_date,
                    error_message="Invalid date format"
                )
            ],
            
            FieldType.NATIONALITY: [
                ValidationRule(
                    field_type=FieldType.NATIONALITY,
                    rule_name="nationality_format",
                    custom_validator=self._validate_nationality,
                    error_message="Invalid nationality"
                )
            ],
            
            FieldType.GENDER: [
                ValidationRule(
                    field_type=FieldType.GENDER,
                    rule_name="gender_format",
                    pattern=r'^(MALE|FEMALE|M|F|LELAKI|PEREMPUAN)$',
                    custom_validator=self._validate_gender,
                    error_message="Invalid gender format"
                )
            ]
        }
        
        return rules
    
    def _validate_field(self, 
                       field_name: str, 
                       value: str, 
                       field_type: FieldType, 
                       document_type: DocumentType) -> ValidationResult:
        """Validate individual field"""
        try:
            # Initialize result
            result = ValidationResult(
                field_name=field_name,
                field_type=field_type,
                original_value=value,
                cleaned_value=value,
                is_valid=True,
                confidence=1.0
            )
            
            # Clean the value
            cleaned_value = self._clean_field_value(value, field_type)
            result.cleaned_value = cleaned_value
            
            # Get validation rules for this field type
            rules = self.validation_rules.get(field_type, [])
            
            # Apply each validation rule
            for rule in rules:
                rule_result = self._apply_validation_rule(cleaned_value, rule, document_type)
                
                if not rule_result['valid']:
                    result.is_valid = False
                    if rule.severity == 'error':
                        result.errors.append(rule_result['message'])
                    elif rule.severity == 'warning':
                        result.warnings.append(rule_result['message'])
                
                # Adjust confidence based on rule results
                if 'confidence_impact' in rule_result:
                    result.confidence *= rule_result['confidence_impact']
            
            # Apply field-specific corrections
            corrected_value, corrections = self._apply_field_corrections(
                cleaned_value, 
                field_type, 
                document_type
            )
            
            if corrections:
                result.cleaned_value = corrected_value
                result.applied_corrections = corrections
            
            return result
            
        except Exception as e:
            logger.error(f"Field validation error for {field_name}: {e}")
            return ValidationResult(
                field_name=field_name,
                field_type=field_type,
                original_value=value,
                cleaned_value=value,
                is_valid=False,
                confidence=0.0,
                errors=[f"Validation error: {str(e)}"]
            )
    
    def _apply_validation_rule(self, 
                             value: str, 
                             rule: ValidationRule, 
                             document_type: DocumentType) -> Dict[str, Any]:
        """Apply a single validation rule"""
        try:
            # Length validation
            if rule.min_length and len(value) < rule.min_length:
                return {
                    'valid': False,
                    'message': f"Value too short (minimum {rule.min_length} characters)",
                    'confidence_impact': 0.5
                }
            
            if rule.max_length and len(value) > rule.max_length:
                return {
                    'valid': False,
                    'message': f"Value too long (maximum {rule.max_length} characters)",
                    'confidence_impact': 0.5
                }
            
            # Pattern validation
            if rule.pattern and not re.match(rule.pattern, value, re.IGNORECASE):
                return {
                    'valid': False,
                    'message': rule.error_message or f"Value doesn't match required pattern",
                    'confidence_impact': 0.3
                }
            
            # Custom validation
            if rule.custom_validator:
                custom_result = rule.custom_validator(value, document_type)
                if not custom_result.get('valid', True):
                    return {
                        'valid': False,
                        'message': custom_result.get('message', rule.error_message),
                        'confidence_impact': custom_result.get('confidence_impact', 0.3)
                    }
            
            return {'valid': True, 'confidence_impact': 1.0}
            
        except Exception as e:
            logger.error(f"Rule application error: {e}")
            return {
                'valid': False,
                'message': f"Rule validation error: {str(e)}",
                'confidence_impact': 0.1
            }
    
    def _validate_malaysian_ic(self, ic_number: str, document_type: DocumentType) -> Dict[str, Any]:
        """Validate Malaysian IC number format and checksum"""
        try:
            # Remove any separators and clean
            clean_ic = re.sub(r'[-\s]', '', ic_number)
            
            if len(clean_ic) != 12:
                return {
                    'valid': False,
                    'message': "IC number must be 12 digits",
                    'confidence_impact': 0.2
                }
            
            # Extract components
            birth_date = clean_ic[:6]
            state_code = clean_ic[6:8]
            sequence = clean_ic[8:]
            
            # Validate birth date
            try:
                year = int(birth_date[:2])
                month = int(birth_date[2:4])
                day = int(birth_date[4:6])
                
                # Determine century (00-30 = 2000s, 31-99 = 1900s)
                if year <= 30:
                    full_year = 2000 + year
                else:
                    full_year = 1900 + year
                
                # Validate date
                birth_date_obj = date(full_year, month, day)
                
                # Check if date is reasonable (not in future, not too old)
                today = date.today()
                if birth_date_obj > today:
                    return {
                        'valid': False,
                        'message': "Birth date cannot be in the future",
                        'confidence_impact': 0.1
                    }
                
                age = today.year - birth_date_obj.year
                if age > 150:
                    return {
                        'valid': False,
                        'message': "Birth date indicates unrealistic age",
                        'confidence_impact': 0.3
                    }
                
            except ValueError:
                return {
                    'valid': False,
                    'message': "Invalid birth date in IC number",
                    'confidence_impact': 0.2
                }
            
            # Validate state code
            if state_code not in self.malaysian_states:
                return {
                    'valid': False,
                    'message': f"Invalid state code: {state_code}",
                    'confidence_impact': 0.4
                }
            
            return {'valid': True, 'confidence_impact': 1.0}
            
        except Exception as e:
            return {
                'valid': False,
                'message': f"IC validation error: {str(e)}",
                'confidence_impact': 0.1
            }
    
    def _validate_malaysian_passport(self, passport_number: str, document_type: DocumentType) -> Dict[str, Any]:
        """Validate Malaysian passport number format"""
        try:
            if not re.match(r'^[A-Z]\d{8}$', passport_number):
                return {
                    'valid': False,
                    'message': "Malaysian passport must start with letter followed by 8 digits",
                    'confidence_impact': 0.3
                }
            
            # Check if first letter is valid for Malaysian passports
            valid_prefixes = ['A', 'H', 'K']  # Common Malaysian passport prefixes
            if passport_number[0] not in valid_prefixes:
                return {
                    'valid': True,  # Still valid but with warning
                    'message': f"Unusual passport prefix: {passport_number[0]}",
                    'confidence_impact': 0.8
                }
            
            return {'valid': True, 'confidence_impact': 1.0}
            
        except Exception as e:
            return {
                'valid': False,
                'message': f"Passport validation error: {str(e)}",
                'confidence_impact': 0.1
            }
    
    def _validate_name(self, name: str, document_type: DocumentType) -> Dict[str, Any]:
        """Validate name format and characters"""
        try:
            # Check for minimum reasonable length
            if len(name.strip()) < 2:
                return {
                    'valid': False,
                    'message': "Name too short",
                    'confidence_impact': 0.2
                }
            
            # Check for valid characters (letters, spaces, hyphens, apostrophes, dots)
            if not re.match(r'^[A-Za-z\s\.\-\']+$', name):
                return {
                    'valid': False,
                    'message': "Name contains invalid characters",
                    'confidence_impact': 0.3
                }
            
            # Check for reasonable word count (1-5 names)
            words = name.strip().split()
            if len(words) > 5:
                return {
                    'valid': False,
                    'message': "Too many name components",
                    'confidence_impact': 0.4
                }
            
            # Check for suspicious patterns
            if re.search(r'(.)\1{3,}', name):  # Repeated characters
                return {
                    'valid': False,
                    'message': "Name contains suspicious repeated characters",
                    'confidence_impact': 0.3
                }
            
            return {'valid': True, 'confidence_impact': 1.0}
            
        except Exception as e:
            return {
                'valid': False,
                'message': f"Name validation error: {str(e)}",
                'confidence_impact': 0.1
            }
    
    def _validate_date(self, date_str: str, document_type: DocumentType) -> Dict[str, Any]:
        """Validate date format and reasonableness"""
        try:
            # Try multiple date formats
            date_formats = [
                '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
                '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
                '%d %m %Y', '%d %B %Y', '%B %d, %Y'
            ]
            
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt).date()
                    break
                except ValueError:
                    continue
            
            if not parsed_date:
                return {
                    'valid': False,
                    'message': "Unrecognized date format",
                    'confidence_impact': 0.2
                }
            
            # Check date reasonableness
            today = date.today()
            
            # For birth dates
            if parsed_date > today:
                return {
                    'valid': False,
                    'message': "Date cannot be in the future",
                    'confidence_impact': 0.1
                }
            
            # Check for unrealistic old dates
            if (today - parsed_date).days > 150 * 365:  # 150 years
                return {
                    'valid': False,
                    'message': "Date is unrealistically old",
                    'confidence_impact': 0.3
                }
            
            return {'valid': True, 'confidence_impact': 1.0}
            
        except Exception as e:
            return {
                'valid': False,
                'message': f"Date validation error: {str(e)}",
                'confidence_impact': 0.1
            }
    
    def _validate_nationality(self, nationality: str, document_type: DocumentType) -> Dict[str, Any]:
        """Validate nationality format"""
        try:
            # Common nationality formats
            valid_nationalities = [
                'MALAYSIAN', 'MALAYSIA', 'WARGANEGARA MALAYSIA',
                'AMERICAN', 'BRITISH', 'CHINESE', 'INDIAN', 'INDONESIAN',
                'THAI', 'SINGAPOREAN', 'FILIPINO', 'VIETNAMESE'
            ]
            
            nationality_upper = nationality.upper().strip()
            
            # Check against known nationalities
            if nationality_upper in valid_nationalities:
                return {'valid': True, 'confidence_impact': 1.0}
            
            # Check if it's a reasonable nationality format
            if re.match(r'^[A-Za-z\s]+$', nationality) and len(nationality) >= 3:
                return {
                    'valid': True,
                    'message': f"Uncommon nationality: {nationality}",
                    'confidence_impact': 0.7
                }
            
            return {
                'valid': False,
                'message': "Invalid nationality format",
                'confidence_impact': 0.3
            }
            
        except Exception as e:
            return {
                'valid': False,
                'message': f"Nationality validation error: {str(e)}",
                'confidence_impact': 0.1
            }
    
    def _validate_gender(self, gender: str, document_type: DocumentType) -> Dict[str, Any]:
        """Validate gender format"""
        try:
            valid_genders = {
                'MALE': ['MALE', 'M', 'LELAKI'],
                'FEMALE': ['FEMALE', 'F', 'PEREMPUAN']
            }
            
            gender_upper = gender.upper().strip()
            
            for standard_gender, variations in valid_genders.items():
                if gender_upper in variations:
                    return {'valid': True, 'confidence_impact': 1.0}
            
            return {
                'valid': False,
                'message': f"Invalid gender: {gender}",
                'confidence_impact': 0.2
            }
            
        except Exception as e:
            return {
                'valid': False,
                'message': f"Gender validation error: {str(e)}",
                'confidence_impact': 0.1
            }
    
    def _validate_cross_field_consistency(self, 
                                        field_results: Dict[str, ValidationResult],
                                        document_type: DocumentType) -> List[str]:
        """Validate consistency between related fields"""
        errors = []
        
        try:
            # Extract cleaned values for cross-validation
            values = {name: result.cleaned_value for name, result in field_results.items()}
            
            # IC number and birth date consistency
            if 'ic_number' in values and 'date_of_birth' in values:
                ic_birth_date = self._extract_birth_date_from_ic(values['ic_number'])
                if ic_birth_date:
                    dob_from_field = self._parse_date(values['date_of_birth'])
                    if dob_from_field and ic_birth_date != dob_from_field:
                        errors.append("Birth date doesn't match IC number birth date")
            
            # IC number and gender consistency
            if 'ic_number' in values and 'gender' in values:
                ic_gender = self._extract_gender_from_ic(values['ic_number'])
                field_gender = self._normalize_gender(values['gender'])
                if ic_gender and field_gender and ic_gender != field_gender:
                    errors.append("Gender doesn't match IC number gender indicator")
            
            # Document dates consistency (issue < expiry)
            if 'issue_date' in values and 'expiry_date' in values:
                issue_date = self._parse_date(values['issue_date'])
                expiry_date = self._parse_date(values['expiry_date'])
                if issue_date and expiry_date and issue_date >= expiry_date:
                    errors.append("Issue date must be before expiry date")
            
            # Age reasonableness check
            if 'date_of_birth' in values:
                birth_date = self._parse_date(values['date_of_birth'])
                if birth_date:
                    age = (date.today() - birth_date).days // 365
                    if age < 0 or age > 150:
                        errors.append(f"Unrealistic age: {age} years")
            
            return errors
            
        except Exception as e:
            logger.error(f"Cross-field validation error: {e}")
            return [f"Cross-field validation error: {str(e)}"]
    
    def _perform_security_checks(self, 
                               field_results: Dict[str, ValidationResult],
                               document_type: DocumentType) -> Dict[str, bool]:
        """Perform security and authenticity checks"""
        checks = {
            'format_consistency': True,
            'data_integrity': True,
            'suspicious_patterns': True,
            'completeness': True
        }
        
        try:
            values = {name: result.cleaned_value for name, result in field_results.items()}
            
            # Check for suspicious patterns
            for field_name, value in values.items():
                # Check for repeated patterns
                if re.search(r'(.{3,})\1{2,}', value):
                    checks['suspicious_patterns'] = False
                
                # Check for keyboard patterns
                if re.search(r'(123|abc|qwe|asd)', value.lower()):
                    checks['suspicious_patterns'] = False
            
            # Check data completeness for document type
            required_fields = self._get_required_fields(document_type)
            missing_fields = [field for field in required_fields if field not in values or not values[field]]
            
            if missing_fields:
                checks['completeness'] = False
            
            # Check format consistency
            error_count = sum(1 for result in field_results.values() if not result.is_valid)
            if error_count > len(field_results) * 0.3:  # More than 30% errors
                checks['format_consistency'] = False
            
            return checks
            
        except Exception as e:
            logger.error(f"Security checks error: {e}")
            return {key: False for key in checks.keys()}
    
    # Helper methods
    def _detect_document_type(self, extracted_data: Dict[str, Any]) -> DocumentType:
        """Auto-detect document type from extracted data"""
        try:
            # Check for IC number pattern
            for key, value in extracted_data.items():
                if 'ic' in key.lower() and value:
                    if re.match(r'^\d{6}-?\d{2}-?\d{4}$', str(value)):
                        return DocumentType.MALAYSIAN_IC
            
            # Check for passport number pattern
            for key, value in extracted_data.items():
                if 'passport' in key.lower() and value:
                    if re.match(r'^[A-Z]\d{8}$', str(value)):
                        return DocumentType.MALAYSIAN_PASSPORT
                    elif re.match(r'^[A-Z0-9]{6,12}$', str(value)):
                        return DocumentType.FOREIGN_PASSPORT
            
            return DocumentType.UNKNOWN
            
        except Exception:
            return DocumentType.UNKNOWN
    
    def _map_field_name_to_type(self, field_name: str) -> Optional[FieldType]:
        """Map field name to field type"""
        field_mapping = {
            'name': FieldType.NAME,
            'full_name': FieldType.NAME,
            'ic_number': FieldType.IC_NUMBER,
            'ic': FieldType.IC_NUMBER,
            'passport_number': FieldType.PASSPORT_NUMBER,
            'passport': FieldType.PASSPORT_NUMBER,
            'date_of_birth': FieldType.DATE_OF_BIRTH,
            'dob': FieldType.DATE_OF_BIRTH,
            'birth_date': FieldType.DATE_OF_BIRTH,
            'nationality': FieldType.NATIONALITY,
            'gender': FieldType.GENDER,
            'sex': FieldType.GENDER,
            'issue_date': FieldType.ISSUE_DATE,
            'expiry_date': FieldType.EXPIRY_DATE,
            'issuing_authority': FieldType.ISSUING_AUTHORITY
        }
        
        return field_mapping.get(field_name.lower())
    
    def _clean_field_value(self, value: str, field_type: FieldType) -> str:
        """Clean field value based on type"""
        try:
            cleaned = value.strip()
            
            if field_type == FieldType.IC_NUMBER:
                # Standardize IC format
                cleaned = re.sub(r'[^\d]', '', cleaned)
                if len(cleaned) == 12:
                    cleaned = f"{cleaned[:6]}-{cleaned[6:8]}-{cleaned[8:]}"
            
            elif field_type == FieldType.NAME:
                # Capitalize properly
                cleaned = ' '.join(word.capitalize() for word in cleaned.split())
            
            elif field_type == FieldType.GENDER:
                # Normalize gender
                cleaned = cleaned.upper()
                if cleaned in ['M', 'MALE', 'LELAKI']:
                    cleaned = 'MALE'
                elif cleaned in ['F', 'FEMALE', 'PEREMPUAN']:
                    cleaned = 'FEMALE'
            
            return cleaned
            
        except Exception:
            return value
    
    def _apply_field_corrections(self, 
                               value: str, 
                               field_type: FieldType, 
                               document_type: DocumentType) -> Tuple[str, List[str]]:
        """Apply automatic corrections to field values"""
        corrected_value = value
        corrections = []
        
        try:
            if field_type == FieldType.IC_NUMBER:
                # Fix common IC format issues
                digits_only = re.sub(r'[^\d]', '', value)
                if len(digits_only) == 12:
                    formatted = f"{digits_only[:6]}-{digits_only[6:8]}-{digits_only[8:]}"
                    if formatted != value:
                        corrected_value = formatted
                        corrections.append("Formatted IC number")
            
            elif field_type == FieldType.PASSPORT_NUMBER:
                # Fix passport format
                if re.match(r'^[a-z]\d{8}$', value.lower()):
                    corrected_value = value.upper()
                    corrections.append("Capitalized passport number")
            
            return corrected_value, corrections
            
        except Exception:
            return value, []
    
    def _calculate_overall_validity(self, 
                                  field_results: Dict[str, ValidationResult],
                                  cross_field_errors: List[str],
                                  security_checks: Dict[str, bool],
                                  confidence_threshold: float) -> Tuple[bool, float]:
        """Calculate overall document validity and confidence"""
        try:
            if not field_results:
                return False, 0.0
            
            # Calculate field-based confidence
            field_confidences = [result.confidence for result in field_results.values()]
            avg_field_confidence = sum(field_confidences) / len(field_confidences)
            
            # Penalty for cross-field errors
            cross_field_penalty = len(cross_field_errors) * 0.1
            
            # Penalty for security check failures
            security_penalty = sum(1 for passed in security_checks.values() if not passed) * 0.15
            
            # Calculate final confidence
            final_confidence = max(0.0, avg_field_confidence - cross_field_penalty - security_penalty)
            
            # Determine validity based on validation level
            if self.validation_level == ValidationLevel.STRICT:
                is_valid = (final_confidence >= confidence_threshold and 
                           len(cross_field_errors) == 0 and 
                           all(security_checks.values()))
            elif self.validation_level == ValidationLevel.MODERATE:
                is_valid = (final_confidence >= confidence_threshold * 0.8 and 
                           len(cross_field_errors) <= 1)
            else:  # LENIENT
                is_valid = final_confidence >= confidence_threshold * 0.6
            
            return is_valid, final_confidence
            
        except Exception as e:
            logger.error(f"Overall validity calculation error: {e}")
            return False, 0.0
    
    def _generate_recommendations(self, 
                                field_results: Dict[str, ValidationResult],
                                cross_field_errors: List[str],
                                security_checks: Dict[str, bool]) -> List[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []
        
        try:
            # Field-specific recommendations
            for field_name, result in field_results.items():
                if not result.is_valid:
                    recommendations.append(f"Fix {field_name}: {', '.join(result.errors)}")
                elif result.warnings:
                    recommendations.append(f"Review {field_name}: {', '.join(result.warnings)}")
            
            # Cross-field recommendations
            if cross_field_errors:
                recommendations.append("Verify consistency between related fields")
            
            # Security recommendations
            if not security_checks.get('completeness', True):
                recommendations.append("Provide missing required fields")
            
            if not security_checks.get('suspicious_patterns', True):
                recommendations.append("Review data for suspicious patterns")
            
            return recommendations
            
        except Exception:
            return ["Review document data quality"]
    
    def _load_malaysian_states(self) -> Dict[str, str]:
        """Load Malaysian state codes"""
        return {
            '01': 'Johor', '02': 'Kedah', '03': 'Kelantan', '04': 'Melaka',
            '05': 'Negeri Sembilan', '06': 'Pahang', '07': 'Pulau Pinang',
            '08': 'Perak', '09': 'Perlis', '10': 'Selangor', '11': 'Terengganu',
            '12': 'Sabah', '13': 'Sarawak', '14': 'Wilayah Persekutuan Kuala Lumpur',
            '15': 'Wilayah Persekutuan Labuan', '16': 'Wilayah Persekutuan Putrajaya'
        }
    
    def _load_country_codes(self) -> Dict[str, str]:
        """Load country codes for nationality validation"""
        return {
            'MY': 'Malaysia', 'US': 'United States', 'GB': 'United Kingdom',
            'CN': 'China', 'IN': 'India', 'ID': 'Indonesia', 'TH': 'Thailand',
            'SG': 'Singapore', 'PH': 'Philippines', 'VN': 'Vietnam'
        }
    
    def _get_required_fields(self, document_type: DocumentType) -> List[str]:
        """Get required fields for document type"""
        if document_type == DocumentType.MALAYSIAN_IC:
            return ['name', 'ic_number', 'date_of_birth', 'gender']
        elif document_type in [DocumentType.MALAYSIAN_PASSPORT, DocumentType.FOREIGN_PASSPORT]:
            return ['name', 'passport_number', 'date_of_birth', 'nationality', 'expiry_date']
        else:
            return ['name']
    
    def _extract_birth_date_from_ic(self, ic_number: str) -> Optional[date]:
        """Extract birth date from Malaysian IC number"""
        try:
            clean_ic = re.sub(r'[-\s]', '', ic_number)
            if len(clean_ic) != 12:
                return None
            
            year = int(clean_ic[:2])
            month = int(clean_ic[2:4])
            day = int(clean_ic[4:6])
            
            # Determine century
            if year <= 30:
                full_year = 2000 + year
            else:
                full_year = 1900 + year
            
            return date(full_year, month, day)
            
        except Exception:
            return None
    
    def _extract_gender_from_ic(self, ic_number: str) -> Optional[str]:
        """Extract gender from Malaysian IC number"""
        try:
            clean_ic = re.sub(r'[-\s]', '', ic_number)
            if len(clean_ic) != 12:
                return None
            
            last_digit = int(clean_ic[-1])
            return 'MALE' if last_digit % 2 == 1 else 'FEMALE'
            
        except Exception:
            return None
    
    def _normalize_gender(self, gender: str) -> Optional[str]:
        """Normalize gender to standard format"""
        gender_upper = gender.upper().strip()
        if gender_upper in ['M', 'MALE', 'LELAKI']:
            return 'MALE'
        elif gender_upper in ['F', 'FEMALE', 'PEREMPUAN']:
            return 'FEMALE'
        return None
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string to date object"""
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
            '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
            '%d %m %Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        
        return None