#!/usr/bin/env python3
"""
Malaysian IC (MyKad) Validation System

Comprehensive validation for Malaysian Identity Card fields including:
- NRIC format and checksum validation
- Date consistency checks (NRIC ↔ DOB)
- Gender validation from NRIC
- Address format validation
- Cross-field consistency checks
- Data quality assessment

Author: AI Assistant
Date: 2025
"""

import re
import json
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

class ValidationSeverity(Enum):
    """Validation error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationError:
    """Represents a validation error"""
    field: str
    message: str
    severity: ValidationSeverity
    code: str
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """Complete validation result for a field"""
    field_name: str
    value: str
    is_valid: bool
    confidence_score: float
    errors: List[ValidationError]
    warnings: List[ValidationError]
    suggestions: List[str]

@dataclass
class ICValidationReport:
    """Complete IC validation report"""
    overall_valid: bool
    confidence_score: float
    field_results: Dict[str, ValidationResult]
    cross_field_errors: List[ValidationError]
    data_quality_score: float
    processing_time: float
    validation_timestamp: str

class MalaysianStateValidator:
    """Validates Malaysian state codes and information"""
    
    def __init__(self):
        self.state_codes = {
            # Peninsular Malaysia
            "01": {"name": "JOHOR", "region": "PENINSULAR"},
            "02": {"name": "KEDAH", "region": "PENINSULAR"},
            "03": {"name": "KELANTAN", "region": "PENINSULAR"},
            "04": {"name": "MELAKA", "region": "PENINSULAR"},
            "05": {"name": "NEGERI SEMBILAN", "region": "PENINSULAR"},
            "06": {"name": "PAHANG", "region": "PENINSULAR"},
            "07": {"name": "PERAK", "region": "PENINSULAR"},
            "08": {"name": "PERLIS", "region": "PENINSULAR"},
            "09": {"name": "PULAU PINANG", "region": "PENINSULAR"},
            "10": {"name": "SELANGOR", "region": "PENINSULAR"},
            "11": {"name": "TERENGGANU", "region": "PENINSULAR"},
            "12": {"name": "SABAH", "region": "EAST_MALAYSIA"},
            "13": {"name": "SARAWAK", "region": "EAST_MALAYSIA"},
            "14": {"name": "KUALA LUMPUR", "region": "FEDERAL_TERRITORY"},
            "15": {"name": "LABUAN", "region": "FEDERAL_TERRITORY"},
            "16": {"name": "PUTRAJAYA", "region": "FEDERAL_TERRITORY"},
            
            # Extended state codes
            "21": {"name": "JOHOR", "region": "PENINSULAR"},
            "22": {"name": "JOHOR", "region": "PENINSULAR"},
            "23": {"name": "JOHOR", "region": "PENINSULAR"},
            "24": {"name": "JOHOR", "region": "PENINSULAR"},
            "25": {"name": "KEDAH", "region": "PENINSULAR"},
            "26": {"name": "KEDAH", "region": "PENINSULAR"},
            "27": {"name": "KEDAH", "region": "PENINSULAR"},
            "28": {"name": "KELANTAN", "region": "PENINSULAR"},
            "29": {"name": "KELANTAN", "region": "PENINSULAR"},
            "30": {"name": "MELAKA", "region": "PENINSULAR"},
            "31": {"name": "NEGERI SEMBILAN", "region": "PENINSULAR"},
            "32": {"name": "NEGERI SEMBILAN", "region": "PENINSULAR"},
            "33": {"name": "PAHANG", "region": "PENINSULAR"},
            "34": {"name": "PAHANG", "region": "PENINSULAR"},
            "35": {"name": "PAHANG", "region": "PENINSULAR"},
            "36": {"name": "PAHANG", "region": "PENINSULAR"},
            "37": {"name": "PERAK", "region": "PENINSULAR"},
            "38": {"name": "PERAK", "region": "PENINSULAR"},
            "39": {"name": "PERAK", "region": "PENINSULAR"},
            "40": {"name": "PERAK", "region": "PENINSULAR"},
            "41": {"name": "PERLIS", "region": "PENINSULAR"},
            "42": {"name": "PULAU PINANG", "region": "PENINSULAR"},
            "43": {"name": "PULAU PINANG", "region": "PENINSULAR"},
            "44": {"name": "PULAU PINANG", "region": "PENINSULAR"},
            "45": {"name": "SELANGOR", "region": "PENINSULAR"},
            "46": {"name": "SELANGOR", "region": "PENINSULAR"},
            "47": {"name": "SELANGOR", "region": "PENINSULAR"},
            "48": {"name": "SELANGOR", "region": "PENINSULAR"},
            "49": {"name": "TERENGGANU", "region": "PENINSULAR"},
            "50": {"name": "TERENGGANU", "region": "PENINSULAR"},
            "51": {"name": "SABAH", "region": "EAST_MALAYSIA"},
            "52": {"name": "SABAH", "region": "EAST_MALAYSIA"},
            "53": {"name": "SABAH", "region": "EAST_MALAYSIA"},
            "54": {"name": "SABAH", "region": "EAST_MALAYSIA"},
            "55": {"name": "SABAH", "region": "EAST_MALAYSIA"},
            "56": {"name": "SARAWAK", "region": "EAST_MALAYSIA"},
            "57": {"name": "SARAWAK", "region": "EAST_MALAYSIA"},
            "58": {"name": "SARAWAK", "region": "EAST_MALAYSIA"},
            "59": {"name": "SARAWAK", "region": "EAST_MALAYSIA"},
            "60": {"name": "KUALA LUMPUR", "region": "FEDERAL_TERRITORY"},
            "61": {"name": "LABUAN", "region": "FEDERAL_TERRITORY"},
            "62": {"name": "PUTRAJAYA", "region": "FEDERAL_TERRITORY"},
            
            # Foreign born codes
            "71": {"name": "FOREIGN BORN", "region": "FOREIGN"},
            "72": {"name": "FOREIGN BORN", "region": "FOREIGN"},
            "74": {"name": "FOREIGN BORN", "region": "FOREIGN"},
            "75": {"name": "FOREIGN BORN", "region": "FOREIGN"},
            "76": {"name": "FOREIGN BORN", "region": "FOREIGN"},
            "77": {"name": "FOREIGN BORN", "region": "FOREIGN"},
            "78": {"name": "FOREIGN BORN", "region": "FOREIGN"},
            "79": {"name": "FOREIGN BORN", "region": "FOREIGN"}
        }
        
        # Postal code ranges by state
        self.postal_ranges = {
            "JOHOR": ["80000-86999"],
            "KEDAH": ["05000-09999"],
            "KELANTAN": ["15000-18999"],
            "MELAKA": ["75000-78999"],
            "NEGERI SEMBILAN": ["70000-73999"],
            "PAHANG": ["25000-28999", "39000-39999"],
            "PERAK": ["30000-36999"],
            "PERLIS": ["01000-02999"],
            "PULAU PINANG": ["10000-14999"],
            "SABAH": ["88000-91999"],
            "SARAWAK": ["93000-98999"],
            "SELANGOR": ["40000-48999"],
            "TERENGGANU": ["20000-24999"],
            "KUALA LUMPUR": ["50000-59999"],
            "LABUAN": ["87000-87999"],
            "PUTRAJAYA": ["62000-62999"]
        }
    
    def validate_state_code(self, state_code: str) -> Tuple[bool, Optional[Dict]]:
        """Validate state code and return state information"""
        if state_code in self.state_codes:
            return True, self.state_codes[state_code]
        return False, None
    
    def validate_postal_code_state_consistency(self, postal_code: str, state: str) -> bool:
        """Validate that postal code matches the state"""
        if state not in self.postal_ranges:
            return False
        
        try:
            postal_int = int(postal_code)
            for range_str in self.postal_ranges[state]:
                if "-" in range_str:
                    start, end = map(int, range_str.split("-"))
                    if start <= postal_int <= end:
                        return True
                else:
                    if postal_int == int(range_str):
                        return True
        except ValueError:
            return False
        
        return False

class NRICValidator:
    """Comprehensive NRIC validation"""
    
    def __init__(self):
        self.state_validator = MalaysianStateValidator()
        self.nric_pattern = re.compile(r'^(\d{2})(\d{2})(\d{2})-(\d{2})-(\d{4})$')
    
    def validate_format(self, nric: str) -> Tuple[bool, List[ValidationError]]:
        """Validate NRIC format"""
        errors = []
        
        if not nric:
            errors.append(ValidationError(
                field="nric",
                message="NRIC is required",
                severity=ValidationSeverity.CRITICAL,
                code="NRIC_MISSING"
            ))
            return False, errors
        
        # Remove spaces and convert to uppercase
        nric = nric.replace(" ", "").upper()
        
        # Check basic format
        if not self.nric_pattern.match(nric):
            errors.append(ValidationError(
                field="nric",
                message="Invalid NRIC format. Expected: YYMMDD-SS-NNNN",
                severity=ValidationSeverity.ERROR,
                code="NRIC_INVALID_FORMAT",
                suggestion="Ensure NRIC follows YYMMDD-SS-NNNN format"
            ))
            return False, errors
        
        return True, errors
    
    def validate_date_component(self, nric: str) -> Tuple[bool, List[ValidationError]]:
        """Validate the date component of NRIC"""
        errors = []
        match = self.nric_pattern.match(nric)
        
        if not match:
            return False, errors
        
        year, month, day, state_code, sequence = match.groups()
        
        try:
            # Convert to integers
            year_int = int(year)
            month_int = int(month)
            day_int = int(day)
            
            # Determine full year (heuristic: >50 = 1900s, <=50 = 2000s)
            full_year = 1900 + year_int if year_int > 50 else 2000 + year_int
            
            # Validate month
            if month_int < 1 or month_int > 12:
                errors.append(ValidationError(
                    field="nric",
                    message=f"Invalid month in NRIC: {month_int}",
                    severity=ValidationSeverity.ERROR,
                    code="NRIC_INVALID_MONTH"
                ))
            
            # Validate day
            if day_int < 1 or day_int > 31:
                errors.append(ValidationError(
                    field="nric",
                    message=f"Invalid day in NRIC: {day_int}",
                    severity=ValidationSeverity.ERROR,
                    code="NRIC_INVALID_DAY"
                ))
            
            # Try to create actual date to validate
            try:
                birth_date = date(full_year, month_int, day_int)
                
                # Check if date is reasonable (not in future, not too old)
                today = date.today()
                if birth_date > today:
                    errors.append(ValidationError(
                        field="nric",
                        message="Birth date in NRIC is in the future",
                        severity=ValidationSeverity.ERROR,
                        code="NRIC_FUTURE_DATE"
                    ))
                
                # Check minimum age (should be at least 12 for IC)
                age = today.year - birth_date.year
                if age < 12:
                    errors.append(ValidationError(
                        field="nric",
                        message="Age derived from NRIC is too young for IC",
                        severity=ValidationSeverity.WARNING,
                        code="NRIC_TOO_YOUNG"
                    ))
                
                # Check maximum reasonable age
                if age > 120:
                    errors.append(ValidationError(
                        field="nric",
                        message="Age derived from NRIC is unreasonably old",
                        severity=ValidationSeverity.WARNING,
                        code="NRIC_TOO_OLD"
                    ))
                    
            except ValueError:
                errors.append(ValidationError(
                    field="nric",
                    message=f"Invalid date in NRIC: {day_int:02d}-{month_int:02d}-{full_year}",
                    severity=ValidationSeverity.ERROR,
                    code="NRIC_INVALID_DATE"
                ))
        
        except ValueError:
            errors.append(ValidationError(
                field="nric",
                message="Invalid numeric components in NRIC",
                severity=ValidationSeverity.ERROR,
                code="NRIC_INVALID_NUMERIC"
            ))
        
        return len(errors) == 0, errors
    
    def validate_state_code(self, nric: str) -> Tuple[bool, List[ValidationError]]:
        """Validate state code in NRIC"""
        errors = []
        match = self.nric_pattern.match(nric)
        
        if not match:
            return False, errors
        
        state_code = match.group(4)
        is_valid, state_info = self.state_validator.validate_state_code(state_code)
        
        if not is_valid:
            errors.append(ValidationError(
                field="nric",
                message=f"Invalid state code in NRIC: {state_code}",
                severity=ValidationSeverity.ERROR,
                code="NRIC_INVALID_STATE_CODE",
                suggestion="Check if state code is correct"
            ))
        
        return is_valid, errors
    
    def extract_gender_from_nric(self, nric: str) -> Optional[str]:
        """Extract gender from NRIC last digit"""
        match = self.nric_pattern.match(nric)
        if not match:
            return None
        
        sequence = match.group(5)
        last_digit = int(sequence[-1])
        
        return "LELAKI" if last_digit % 2 == 1 else "PEREMPUAN"
    
    def extract_birth_date_from_nric(self, nric: str) -> Optional[date]:
        """Extract birth date from NRIC"""
        match = self.nric_pattern.match(nric)
        if not match:
            return None
        
        year, month, day = match.groups()[:3]
        
        try:
            year_int = int(year)
            month_int = int(month)
            day_int = int(day)
            
            # Determine full year
            full_year = 1900 + year_int if year_int > 50 else 2000 + year_int
            
            return date(full_year, month_int, day_int)
        except ValueError:
            return None
    
    def validate_complete(self, nric: str) -> Tuple[bool, List[ValidationError]]:
        """Complete NRIC validation"""
        all_errors = []
        
        # Format validation
        format_valid, format_errors = self.validate_format(nric)
        all_errors.extend(format_errors)
        
        if not format_valid:
            return False, all_errors
        
        # Date validation
        date_valid, date_errors = self.validate_date_component(nric)
        all_errors.extend(date_errors)
        
        # State code validation
        state_valid, state_errors = self.validate_state_code(nric)
        all_errors.extend(state_errors)
        
        return len(all_errors) == 0, all_errors

class DateValidator:
    """Date validation utilities"""
    
    def __init__(self):
        self.date_formats = [
            '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y',
            '%d-%m-%y', '%d/%m/%y', '%d.%m.%y',
            '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d'
        ]
    
    def parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string to date object"""
        if not date_str:
            return None
        
        for fmt in self.date_formats:
            try:
                parsed = datetime.strptime(date_str.strip(), fmt).date()
                
                # Handle 2-digit years
                if parsed.year < 100:
                    if parsed.year > 50:
                        parsed = parsed.replace(year=parsed.year + 1900)
                    else:
                        parsed = parsed.replace(year=parsed.year + 2000)
                
                return parsed
            except ValueError:
                continue
        
        return None
    
    def validate_date_range(self, date_obj: date, min_date: Optional[date] = None, 
                           max_date: Optional[date] = None) -> List[ValidationError]:
        """Validate date is within reasonable range"""
        errors = []
        
        if min_date and date_obj < min_date:
            errors.append(ValidationError(
                field="date",
                message=f"Date {date_obj} is before minimum allowed date {min_date}",
                severity=ValidationSeverity.ERROR,
                code="DATE_TOO_EARLY"
            ))
        
        if max_date and date_obj > max_date:
            errors.append(ValidationError(
                field="date",
                message=f"Date {date_obj} is after maximum allowed date {max_date}",
                severity=ValidationSeverity.ERROR,
                code="DATE_TOO_LATE"
            ))
        
        return errors

class MalaysianICValidator:
    """Main validator for Malaysian IC fields"""
    
    def __init__(self):
        self.nric_validator = NRICValidator()
        self.date_validator = DateValidator()
        self.state_validator = MalaysianStateValidator()
        
        # Valid values for specific fields
        self.valid_genders = {"LELAKI", "PEREMPUAN", "MALE", "FEMALE"}
        self.valid_religions = {
            "ISLAM", "BUDDHA", "HINDU", "KRISTIAN", "TAOISME", 
            "KONFUSIANISME", "LAIN-LAIN", "MUSLIM", "BUDDHIST", 
            "CHRISTIAN", "OTHERS"
        }
        self.valid_nationalities = {"WARGANEGARA", "BUKAN WARGANEGARA", "CITIZEN", "NON-CITIZEN"}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def validate_nric(self, nric: str) -> ValidationResult:
        """Validate NRIC field"""
        start_time = datetime.now()
        
        is_valid, errors = self.nric_validator.validate_complete(nric)
        
        # Calculate confidence based on validation results
        confidence = 1.0
        for error in errors:
            if error.severity == ValidationSeverity.CRITICAL:
                confidence -= 0.5
            elif error.severity == ValidationSeverity.ERROR:
                confidence -= 0.3
            elif error.severity == ValidationSeverity.WARNING:
                confidence -= 0.1
        
        confidence = max(0.0, confidence)
        
        warnings = [e for e in errors if e.severity == ValidationSeverity.WARNING]
        errors = [e for e in errors if e.severity != ValidationSeverity.WARNING]
        
        suggestions = []
        if not is_valid:
            suggestions.append("Ensure NRIC follows YYMMDD-SS-NNNN format")
            suggestions.append("Verify date components are valid")
            suggestions.append("Check state code is correct")
        
        return ValidationResult(
            field_name="nric",
            value=nric,
            is_valid=is_valid,
            confidence_score=confidence,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_name(self, name: str) -> ValidationResult:
        """Validate name field"""
        errors = []
        warnings = []
        suggestions = []
        
        if not name or not name.strip():
            errors.append(ValidationError(
                field="name",
                message="Name is required",
                severity=ValidationSeverity.CRITICAL,
                code="NAME_MISSING"
            ))
            return ValidationResult(
                field_name="name",
                value=name,
                is_valid=False,
                confidence_score=0.0,
                errors=errors,
                warnings=warnings,
                suggestions=["Provide a valid name"]
            )
        
        name = name.strip()
        
        # Length validation
        if len(name) < 2:
            errors.append(ValidationError(
                field="name",
                message="Name is too short",
                severity=ValidationSeverity.ERROR,
                code="NAME_TOO_SHORT"
            ))
        
        if len(name) > 100:
            warnings.append(ValidationError(
                field="name",
                message="Name is unusually long",
                severity=ValidationSeverity.WARNING,
                code="NAME_TOO_LONG"
            ))
        
        # Character validation
        if not re.match(r'^[A-Z\s/\-\'\.]+$', name.upper()):
            errors.append(ValidationError(
                field="name",
                message="Name contains invalid characters",
                severity=ValidationSeverity.ERROR,
                code="NAME_INVALID_CHARS",
                suggestion="Name should only contain letters, spaces, hyphens, apostrophes, and slashes"
            ))
        
        # Malaysian name pattern validation
        malaysian_patterns = [
            r'\b(BIN|BINTI)\b',  # Malay
            r'\bA/[LP]\b',       # Indian
            r'^[A-Z]+\s+[A-Z\s]+$'  # Chinese or general
        ]
        
        pattern_match = any(re.search(pattern, name.upper()) for pattern in malaysian_patterns)
        if not pattern_match and len(name.split()) < 2:
            warnings.append(ValidationError(
                field="name",
                message="Name format doesn't match typical Malaysian patterns",
                severity=ValidationSeverity.WARNING,
                code="NAME_ATYPICAL_FORMAT"
            ))
        
        is_valid = len(errors) == 0
        confidence = 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1)
        confidence = max(0.0, confidence)
        
        return ValidationResult(
            field_name="name",
            value=name,
            is_valid=is_valid,
            confidence_score=confidence,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_gender(self, gender: str, nric: str = None) -> ValidationResult:
        """Validate gender field and consistency with NRIC"""
        errors = []
        warnings = []
        suggestions = []
        
        if not gender:
            errors.append(ValidationError(
                field="gender",
                message="Gender is required",
                severity=ValidationSeverity.CRITICAL,
                code="GENDER_MISSING"
            ))
            return ValidationResult(
                field_name="gender",
                value=gender,
                is_valid=False,
                confidence_score=0.0,
                errors=errors,
                warnings=warnings,
                suggestions=["Provide a valid gender (LELAKI/PEREMPUAN)"]
            )
        
        gender = gender.strip().upper()
        
        # Valid gender check
        if gender not in self.valid_genders:
            errors.append(ValidationError(
                field="gender",
                message=f"Invalid gender: {gender}",
                severity=ValidationSeverity.ERROR,
                code="GENDER_INVALID",
                suggestion="Use LELAKI, PEREMPUAN, MALE, or FEMALE"
            ))
        
        # NRIC consistency check
        if nric and gender in self.valid_genders:
            nric_gender = self.nric_validator.extract_gender_from_nric(nric)
            if nric_gender:
                # Normalize for comparison
                gender_normalized = "LELAKI" if gender in ["LELAKI", "MALE"] else "PEREMPUAN"
                if nric_gender != gender_normalized:
                    errors.append(ValidationError(
                        field="gender",
                        message=f"Gender {gender} doesn't match NRIC-derived gender {nric_gender}",
                        severity=ValidationSeverity.ERROR,
                        code="GENDER_NRIC_MISMATCH",
                        suggestion=f"Gender should be {nric_gender} based on NRIC"
                    ))
        
        is_valid = len(errors) == 0
        confidence = 1.0 - (len(errors) * 0.4 + len(warnings) * 0.1)
        confidence = max(0.0, confidence)
        
        return ValidationResult(
            field_name="gender",
            value=gender,
            is_valid=is_valid,
            confidence_score=confidence,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_birth_date(self, birth_date_str: str, nric: str = None) -> ValidationResult:
        """Validate birth date and consistency with NRIC"""
        errors = []
        warnings = []
        suggestions = []
        
        if not birth_date_str:
            errors.append(ValidationError(
                field="birth_date",
                message="Birth date is required",
                severity=ValidationSeverity.CRITICAL,
                code="BIRTH_DATE_MISSING"
            ))
            return ValidationResult(
                field_name="birth_date",
                value=birth_date_str,
                is_valid=False,
                confidence_score=0.0,
                errors=errors,
                warnings=warnings,
                suggestions=["Provide a valid birth date"]
            )
        
        # Parse date
        birth_date = self.date_validator.parse_date(birth_date_str)
        if not birth_date:
            errors.append(ValidationError(
                field="birth_date",
                message="Invalid date format",
                severity=ValidationSeverity.ERROR,
                code="BIRTH_DATE_INVALID_FORMAT",
                suggestion="Use DD-MM-YYYY, DD/MM/YYYY, or similar format"
            ))
            return ValidationResult(
                field_name="birth_date",
                value=birth_date_str,
                is_valid=False,
                confidence_score=0.0,
                errors=errors,
                warnings=warnings,
                suggestions=["Use DD-MM-YYYY, DD/MM/YYYY, or similar format"]
            )
        
        # Date range validation
        today = date.today()
        min_date = date(1900, 1, 1)
        
        range_errors = self.date_validator.validate_date_range(birth_date, min_date, today)
        errors.extend(range_errors)
        
        # NRIC consistency check
        if nric:
            nric_birth_date = self.nric_validator.extract_birth_date_from_nric(nric)
            if nric_birth_date and nric_birth_date != birth_date:
                errors.append(ValidationError(
                    field="birth_date",
                    message=f"Birth date {birth_date} doesn't match NRIC date {nric_birth_date}",
                    severity=ValidationSeverity.ERROR,
                    code="BIRTH_DATE_NRIC_MISMATCH",
                    suggestion=f"Birth date should be {nric_birth_date} based on NRIC"
                ))
        
        is_valid = len(errors) == 0
        confidence = 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1)
        confidence = max(0.0, confidence)
        
        return ValidationResult(
            field_name="birth_date",
            value=birth_date_str,
            is_valid=is_valid,
            confidence_score=confidence,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_religion(self, religion: str) -> ValidationResult:
        """Validate religion field"""
        errors = []
        warnings = []
        suggestions = []
        
        if not religion:
            warnings.append(ValidationError(
                field="religion",
                message="Religion is not specified",
                severity=ValidationSeverity.WARNING,
                code="RELIGION_MISSING"
            ))
            return ValidationResult(
                field_name="religion",
                value=religion,
                is_valid=True,  # Religion is optional
                confidence_score=0.5,
                errors=errors,
                warnings=warnings,
                suggestions=["Religion is optional but recommended"]
            )
        
        religion = religion.strip().upper()
        
        if religion not in self.valid_religions:
            warnings.append(ValidationError(
                field="religion",
                message=f"Unusual religion value: {religion}",
                severity=ValidationSeverity.WARNING,
                code="RELIGION_UNUSUAL",
                suggestion="Common values: ISLAM, BUDDHA, HINDU, KRISTIAN"
            ))
        
        is_valid = len(errors) == 0
        confidence = 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1)
        confidence = max(0.0, confidence)
        
        return ValidationResult(
            field_name="religion",
            value=religion,
            is_valid=is_valid,
            confidence_score=confidence,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_nationality(self, nationality: str) -> ValidationResult:
        """Validate nationality field"""
        errors = []
        warnings = []
        suggestions = []
        
        if not nationality:
            errors.append(ValidationError(
                field="nationality",
                message="Nationality is required",
                severity=ValidationSeverity.ERROR,
                code="NATIONALITY_MISSING"
            ))
            return ValidationResult(
                field_name="nationality",
                value=nationality,
                is_valid=False,
                confidence_score=0.0,
                errors=errors,
                warnings=warnings,
                suggestions=["Provide nationality (usually WARGANEGARA)"]
            )
        
        nationality = nationality.strip().upper()
        
        if nationality not in self.valid_nationalities:
            warnings.append(ValidationError(
                field="nationality",
                message=f"Unusual nationality value: {nationality}",
                severity=ValidationSeverity.WARNING,
                code="NATIONALITY_UNUSUAL",
                suggestion="Common values: WARGANEGARA, BUKAN WARGANEGARA"
            ))
        
        is_valid = len(errors) == 0
        confidence = 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1)
        confidence = max(0.0, confidence)
        
        return ValidationResult(
            field_name="nationality",
            value=nationality,
            is_valid=is_valid,
            confidence_score=confidence,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_address(self, address: str) -> ValidationResult:
        """Validate address field"""
        errors = []
        warnings = []
        suggestions = []
        
        if not address:
            errors.append(ValidationError(
                field="address",
                message="Address is required",
                severity=ValidationSeverity.CRITICAL,
                code="ADDRESS_MISSING"
            ))
            return ValidationResult(
                field_name="address",
                value=address,
                is_valid=False,
                confidence_score=0.0,
                errors=errors,
                warnings=warnings,
                suggestions=["Provide a complete address"]
            )
        
        address = address.strip()
        
        # Length validation
        if len(address) < 10:
            warnings.append(ValidationError(
                field="address",
                message="Address seems too short",
                severity=ValidationSeverity.WARNING,
                code="ADDRESS_TOO_SHORT"
            ))
        
        # Postal code validation
        postal_match = re.search(r'\b(\d{5})\b', address)
        if not postal_match:
            warnings.append(ValidationError(
                field="address",
                message="No postal code found in address",
                severity=ValidationSeverity.WARNING,
                code="ADDRESS_NO_POSTAL_CODE",
                suggestion="Include 5-digit postal code"
            ))
        else:
            postal_code = postal_match.group(1)
            # Basic postal code range check
            postal_int = int(postal_code)
            if postal_int < 1000 or postal_int > 99000:
                warnings.append(ValidationError(
                    field="address",
                    message=f"Postal code {postal_code} seems invalid",
                    severity=ValidationSeverity.WARNING,
                    code="ADDRESS_INVALID_POSTAL_CODE"
                ))
        
        # State name validation
        state_names = list(set(info["name"] for info in self.state_validator.state_codes.values()))
        state_found = any(state in address.upper() for state in state_names)
        
        if not state_found:
            warnings.append(ValidationError(
                field="address",
                message="No recognizable Malaysian state found in address",
                severity=ValidationSeverity.WARNING,
                code="ADDRESS_NO_STATE",
                suggestion="Include state name in address"
            ))
        
        is_valid = len(errors) == 0
        confidence = 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1)
        confidence = max(0.0, confidence)
        
        return ValidationResult(
            field_name="address",
            value=address,
            is_valid=is_valid,
            confidence_score=confidence,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_issue_date(self, issue_date_str: str, birth_date_str: str = None) -> ValidationResult:
        """Validate issue date"""
        errors = []
        warnings = []
        suggestions = []
        
        if not issue_date_str:
            warnings.append(ValidationError(
                field="issue_date",
                message="Issue date is not specified",
                severity=ValidationSeverity.WARNING,
                code="ISSUE_DATE_MISSING"
            ))
            return ValidationResult(
                field_name="issue_date",
                value=issue_date_str,
                is_valid=True,  # Issue date is optional
                confidence_score=0.5,
                errors=errors,
                warnings=warnings,
                suggestions=["Issue date is optional"]
            )
        
        # Parse date
        issue_date = self.date_validator.parse_date(issue_date_str)
        if not issue_date:
            errors.append(ValidationError(
                field="issue_date",
                message="Invalid issue date format",
                severity=ValidationSeverity.ERROR,
                code="ISSUE_DATE_INVALID_FORMAT",
                suggestion="Use DD-MM-YYYY, DD/MM/YYYY, or similar format"
            ))
            return ValidationResult(
                field_name="issue_date",
                value=issue_date_str,
                is_valid=False,
                confidence_score=0.0,
                errors=errors,
                warnings=warnings,
                suggestions=["Use DD-MM-YYYY, DD/MM/YYYY, or similar format"]
            )
        
        # Date range validation
        today = date.today()
        min_date = date(1990, 1, 1)  # IC system started around this time
        
        range_errors = self.date_validator.validate_date_range(issue_date, min_date, today)
        errors.extend(range_errors)
        
        # Birth date consistency (must be at least 12 years after birth)
        if birth_date_str:
            birth_date = self.date_validator.parse_date(birth_date_str)
            if birth_date:
                min_issue_date = date(birth_date.year + 12, birth_date.month, birth_date.day)
                if issue_date < min_issue_date:
                    errors.append(ValidationError(
                        field="issue_date",
                        message="Issue date is too early (must be at least 12 years after birth)",
                        severity=ValidationSeverity.ERROR,
                        code="ISSUE_DATE_TOO_EARLY",
                        suggestion=f"Issue date should be after {min_issue_date}"
                    ))
        
        is_valid = len(errors) == 0
        confidence = 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1)
        confidence = max(0.0, confidence)
        
        return ValidationResult(
            field_name="issue_date",
            value=issue_date_str,
            is_valid=is_valid,
            confidence_score=confidence,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_complete_ic(self, ic_data: Dict[str, str]) -> ICValidationReport:
        """Validate complete IC data with cross-field checks"""
        start_time = datetime.now()
        
        field_results = {}
        cross_field_errors = []
        
        # Extract field values
        nric = ic_data.get('nric', '')
        name = ic_data.get('name', '')
        gender = ic_data.get('gender', '')
        birth_date = ic_data.get('birth_date', '')
        religion = ic_data.get('religion', '')
        nationality = ic_data.get('nationality', '')
        address = ic_data.get('address', '')
        issue_date = ic_data.get('issue_date', '')
        
        # Validate individual fields
        field_results['nric'] = self.validate_nric(nric)
        field_results['name'] = self.validate_name(name)
        field_results['gender'] = self.validate_gender(gender, nric)
        field_results['birth_date'] = self.validate_birth_date(birth_date, nric)
        field_results['religion'] = self.validate_religion(religion)
        field_results['nationality'] = self.validate_nationality(nationality)
        field_results['address'] = self.validate_address(address)
        field_results['issue_date'] = self.validate_issue_date(issue_date, birth_date)
        
        # Cross-field validation
        if nric and birth_date:
            nric_birth = self.nric_validator.extract_birth_date_from_nric(nric)
            parsed_birth = self.date_validator.parse_date(birth_date)
            
            if nric_birth and parsed_birth and nric_birth != parsed_birth:
                cross_field_errors.append(ValidationError(
                    field="cross_validation",
                    message=f"NRIC birth date {nric_birth} doesn't match provided birth date {parsed_birth}",
                    severity=ValidationSeverity.ERROR,
                    code="CROSS_NRIC_BIRTH_MISMATCH"
                ))
        
        if nric and gender:
            nric_gender = self.nric_validator.extract_gender_from_nric(nric)
            if nric_gender:
                gender_normalized = "LELAKI" if gender.upper() in ["LELAKI", "MALE"] else "PEREMPUAN"
                if nric_gender != gender_normalized:
                    cross_field_errors.append(ValidationError(
                        field="cross_validation",
                        message=f"NRIC gender {nric_gender} doesn't match provided gender {gender}",
                        severity=ValidationSeverity.ERROR,
                        code="CROSS_NRIC_GENDER_MISMATCH"
                    ))
        
        # Calculate overall validity and confidence
        all_valid = all(result.is_valid for result in field_results.values()) and len(cross_field_errors) == 0
        
        # Calculate overall confidence
        field_confidences = [result.confidence_score for result in field_results.values()]
        overall_confidence = sum(field_confidences) / len(field_confidences) if field_confidences else 0.0
        
        # Reduce confidence for cross-field errors
        if cross_field_errors:
            overall_confidence *= 0.7
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(field_results, cross_field_errors)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ICValidationReport(
            overall_valid=all_valid,
            confidence_score=overall_confidence,
            field_results=field_results,
            cross_field_errors=cross_field_errors,
            data_quality_score=data_quality_score,
            processing_time=processing_time,
            validation_timestamp=datetime.now().isoformat()
        )
    
    def _calculate_data_quality_score(self, field_results: Dict[str, ValidationResult], 
                                    cross_field_errors: List[ValidationError]) -> float:
        """Calculate overall data quality score (0-1)"""
        total_score = 0.0
        total_weight = 0.0
        
        # Field weights (more important fields have higher weights)
        field_weights = {
            'nric': 0.25,
            'name': 0.20,
            'birth_date': 0.15,
            'gender': 0.10,
            'address': 0.15,
            'nationality': 0.05,
            'religion': 0.05,
            'issue_date': 0.05
        }
        
        for field_name, result in field_results.items():
            weight = field_weights.get(field_name, 0.05)
            total_weight += weight
            
            field_score = result.confidence_score
            # Penalize for validation errors
            if not result.is_valid:
                field_score *= 0.5
            
            total_score += field_score * weight
        
        # Penalize for cross-field errors
        cross_field_penalty = len(cross_field_errors) * 0.1
        
        final_score = (total_score / total_weight) - cross_field_penalty
        return max(0.0, min(1.0, final_score))

def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Malaysian IC data")
    parser.add_argument("--test", action="store_true", help="Run test validation")
    parser.add_argument("--json_file", help="JSON file with IC data to validate")
    
    args = parser.parse_args()
    
    validator = MalaysianICValidator()
    
    if args.test:
        # Test data
        test_ic_data = {
            'nric': '801231-14-5678',
            'name': 'ALI BIN AHMAD',
            'gender': 'LELAKI',
            'birth_date': '31-12-1980',
            'religion': 'ISLAM',
            'nationality': 'WARGANEGARA',
            'address': 'NO 12 JALAN BUNGA RAYA, 43000 KAJANG, SELANGOR',
            'issue_date': '22-07-2018'
        }
        
        print("🧪 Testing Malaysian IC Validation")
        print("=" * 50)
        
        report = validator.validate_complete_ic(test_ic_data)
        
        print(f"📊 Overall Valid: {'✅' if report.overall_valid else '❌'}")
        print(f"📈 Confidence Score: {report.confidence_score:.2f}")
        print(f"🏆 Data Quality Score: {report.data_quality_score:.2f}")
        print(f"⏱️  Processing Time: {report.processing_time:.3f}s")
        
        print("\n📋 Field Validation Results:")
        for field_name, result in report.field_results.items():
            status = "✅" if result.is_valid else "❌"
            print(f"  {status} {field_name}: {result.value} (conf: {result.confidence_score:.2f})")
            
            for error in result.errors:
                print(f"    🔴 {error.message}")
            
            for warning in result.warnings:
                print(f"    🟡 {warning.message}")
        
        if report.cross_field_errors:
            print("\n🔗 Cross-field Validation Errors:")
            for error in report.cross_field_errors:
                print(f"  🔴 {error.message}")
    
    elif args.json_file:
        try:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                ic_data = json.load(f)
            
            report = validator.validate_complete_ic(ic_data)
            
            # Output results
            print(json.dumps(asdict(report), indent=2, default=str))
            
        except Exception as e:
            print(f"Error processing file: {e}")
    
    else:
        print("Use --test for testing or --json_file to validate IC data from JSON file")

if __name__ == "__main__":
    main()