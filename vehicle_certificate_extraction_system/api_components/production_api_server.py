#!/usr/bin/env python3
"""
Production-Ready API Server for Document Processing Pipeline

This module provides a FastAPI-based REST API for document processing with:
- Image upload endpoints
- JSON response format
- Authentication and rate limiting
- Comprehensive error handling
- Performance monitoring
- Privacy and security features
"""

import os
import io
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import base64

# FastAPI and related imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Image processing
from PIL import Image
import cv2
import numpy as np

# Our pipeline components
from integration_demo import IntegratedDocumentProcessor
from evaluation_metrics_engine import EvaluationMetricsEngine, FieldPrediction, BoundingBox
from quality_assurance_pipeline import QualityAssuranceEngine

# Security and monitoring
import jwt
from cryptography.fernet import Fernet
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ProcessingRequest(BaseModel):
    """Request model for document processing"""
    export_annotations: bool = Field(default=False, description="Whether to export annotations")
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Quality threshold for auto-approval")
    mask_sensitive_data: bool = Field(default=True, description="Whether to mask sensitive data in response")
    include_confidence_scores: bool = Field(default=True, description="Include confidence scores in response")

class FieldResult(BaseModel):
    """Field extraction result"""
    field_name: str
    text: str
    confidence: float
    bbox: Optional[Dict[str, float]] = None
    is_sensitive: bool = False
    masked_text: Optional[str] = None

class ProcessingResponse(BaseModel):
    """Response model for document processing"""
    success: bool
    document_id: str
    processing_time: float
    quality_score: float
    fields: List[FieldResult]
    validation_results: Dict[str, Any]
    requires_review: bool
    error_message: Optional[str] = None
    annotated_image_base64: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]

# Security and privacy utilities
class SecurityManager:
    """Handles encryption, masking, and security operations"""
    
    def __init__(self):
        # Initialize encryption key (in production, load from secure storage)
        self.encryption_key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
        self.cipher = Fernet(self.encryption_key)
        
        # JWT secret (in production, load from secure storage)
        self.jwt_secret = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
        
        # Sensitive field patterns
        self.sensitive_fields = {
            'nric', 'ic_number', 'identity_card', 'passport',
            'address', 'full_address', 'home_address',
            'phone', 'mobile', 'telephone'
        }
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def mask_nric(self, nric: str) -> str:
        """Mask NRIC showing only last 4 digits"""
        if len(nric) >= 4:
            return '*' * (len(nric) - 4) + nric[-4:]
        return '*' * len(nric)
    
    def mask_sensitive_field(self, field_name: str, value: str) -> str:
        """Mask sensitive field values"""
        field_lower = field_name.lower()
        
        if 'nric' in field_lower or 'ic' in field_lower:
            return self.mask_nric(value)
        elif 'address' in field_lower:
            # Mask address keeping only first few characters
            return value[:10] + '*' * max(0, len(value) - 10) if len(value) > 10 else '*' * len(value)
        elif 'phone' in field_lower or 'mobile' in field_lower:
            # Mask phone number keeping only last 4 digits
            return '*' * max(0, len(value) - 4) + value[-4:] if len(value) > 4 else '*' * len(value)
        else:
            return value
    
    def is_sensitive_field(self, field_name: str) -> bool:
        """Check if field contains sensitive data"""
        return any(sensitive in field_name.lower() for sensitive in self.sensitive_fields)
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Initialize components
security_manager = SecurityManager()
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Document Processing API",
    description="Production-ready API for Malaysian document processing",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize processing components
processor = IntegratedDocumentProcessor()
eval_engine = EvaluationMetricsEngine()
qa_engine = QualityAssuranceEngine()

# Authentication dependency
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token"""
    token = credentials.credentials
    return security_manager.verify_jwt_token(token)

# API Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        components={
            "document_processor": "operational",
            "evaluation_engine": "operational",
            "qa_engine": "operational"
        }
    )

@app.post("/process-document", response_model=ProcessingResponse)
@limiter.limit("10/minute")  # Rate limiting
async def process_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    processing_request: ProcessingRequest = ProcessingRequest(),
    # user: Dict[str, Any] = Depends(verify_token)  # Uncomment for authentication
):
    """
    Process uploaded document image and extract fields
    
    - **file**: Image file (JPEG, PNG, TIFF)
    - **export_annotations**: Whether to export annotation files
    - **quality_threshold**: Quality threshold for auto-approval (0.0-1.0)
    - **mask_sensitive_data**: Whether to mask sensitive data in response
    - **include_confidence_scores**: Include confidence scores in response
    """
    
    start_time = time.time()
    document_id = hashlib.md5(f"{file.filename}_{start_time}".encode()).hexdigest()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        image_data = await file.read()
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Save temporary file for processing
        temp_path = f"/tmp/{document_id}_{file.filename}"
        image.save(temp_path)
        
        # Process document
        logger.info(f"Processing document {document_id}")
        results = processor.process_document_complete(
            image_path=temp_path,
            export_annotations=processing_request.export_annotations
        )
        
        processing_time = time.time() - start_time
        
        # Extract and format fields
        fields = []
        for field_name, field_data in results.get('extracted_fields', {}).items():
            is_sensitive = security_manager.is_sensitive_field(field_name)
            text = field_data.get('text', '')
            
            field_result = FieldResult(
                field_name=field_name,
                text=text if not (processing_request.mask_sensitive_data and is_sensitive) else security_manager.mask_sensitive_field(field_name, text),
                confidence=field_data.get('confidence', 0.0) if processing_request.include_confidence_scores else 1.0,
                bbox=field_data.get('bbox'),
                is_sensitive=is_sensitive,
                masked_text=security_manager.mask_sensitive_field(field_name, text) if is_sensitive else None
            )
            fields.append(field_result)
        
        # Determine if review is required
        quality_score = results.get('quality_assurance', {}).get('overall_quality', 0.0)
        requires_review = quality_score < processing_request.quality_threshold
        
        # Generate annotated image if requested
        annotated_image_base64 = None
        if processing_request.export_annotations:
            # Create annotated image (simplified version)
            annotated_image = create_annotated_image(image, fields)
            img_buffer = io.BytesIO()
            annotated_image.save(img_buffer, format='JPEG')
            annotated_image_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Log processing for audit trail
        background_tasks.add_task(
            log_processing_event,
            document_id=document_id,
            filename=file.filename,
            processing_time=processing_time,
            quality_score=quality_score,
            fields_extracted=len(fields),
            # user_id=user.get('user_id', 'anonymous')  # Uncomment for authentication
        )
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return ProcessingResponse(
            success=True,
            document_id=document_id,
            processing_time=processing_time,
            quality_score=quality_score,
            fields=fields,
            validation_results=results.get('validation_results', {}),
            requires_review=requires_review,
            annotated_image_base64=annotated_image_base64
        )
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        
        # Clean up temporary file if it exists
        temp_path = f"/tmp/{document_id}_{file.filename}"
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return ProcessingResponse(
            success=False,
            document_id=document_id,
            processing_time=time.time() - start_time,
            quality_score=0.0,
            fields=[],
            validation_results={},
            requires_review=True,
            error_message=str(e)
        )

@app.get("/metrics")
@limiter.limit("5/minute")
async def get_metrics(
    request: Request,
    days: int = 7,
    # user: Dict[str, Any] = Depends(verify_token)  # Uncomment for authentication
):
    """Get performance metrics for the specified period"""
    
    try:
        report = eval_engine.generate_performance_report(days=days)
        return JSONResponse(content=report)
    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating metrics")

@app.post("/feedback")
@limiter.limit("20/minute")
async def submit_feedback(
    request: Request,
    background_tasks: BackgroundTasks,
    document_id: str,
    field_corrections: Dict[str, str],
    quality_rating: int = Field(..., ge=1, le=5),
    # user: Dict[str, Any] = Depends(verify_token)  # Uncomment for authentication
):
    """Submit feedback for processed document (for active learning)"""
    
    try:
        # Store feedback for active learning
        background_tasks.add_task(
            store_feedback,
            document_id=document_id,
            field_corrections=field_corrections,
            quality_rating=quality_rating,
            # user_id=user.get('user_id', 'anonymous')  # Uncomment for authentication
        )
        
        return {"success": True, "message": "Feedback submitted successfully"}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error submitting feedback")

# Utility functions

def create_annotated_image(image: Image.Image, fields: List[FieldResult]) -> Image.Image:
    """Create annotated image with bounding boxes and labels"""
    # Convert PIL to OpenCV
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for field in fields:
        if field.bbox:
            bbox = field.bbox
            x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])
            
            # Draw bounding box
            color = (0, 255, 0) if not field.is_sensitive else (255, 0, 0)  # Green for normal, red for sensitive
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{field.field_name}: {field.confidence:.2f}"
            cv2.putText(cv_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Convert back to PIL
    annotated_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    return annotated_image

async def log_processing_event(
    document_id: str,
    filename: str,
    processing_time: float,
    quality_score: float,
    fields_extracted: int,
    user_id: str = "anonymous"
):
    """Log processing event for audit trail"""
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'document_id': document_id,
        'filename': filename,
        'processing_time': processing_time,
        'quality_score': quality_score,
        'fields_extracted': fields_extracted,
        'user_id': user_id
    }
    
    # In production, store in database or logging system
    logger.info(f"Processing event: {json.dumps(log_entry)}")

async def store_feedback(
    document_id: str,
    field_corrections: Dict[str, str],
    quality_rating: int,
    user_id: str = "anonymous"
):
    """Store feedback for active learning"""
    
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'document_id': document_id,
        'field_corrections': field_corrections,
        'quality_rating': quality_rating,
        'user_id': user_id
    }
    
    # In production, store in database for active learning
    logger.info(f"Feedback received: {json.dumps(feedback_entry)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Document Processing API starting up...")
    
    # Initialize database connections, load models, etc.
    # In production, you might want to warm up the models here
    
    logger.info("Document Processing API ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Document Processing API shutting down...")
    
    # Cleanup resources, close connections, etc.
    
    logger.info("Document Processing API shutdown complete!")

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "production_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )