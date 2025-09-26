# Malaysian Vehicle Certificate Extraction System

A comprehensive, production-ready system for extracting information from Malaysian Vehicle Registration Certificates (Sijil Pemilikan Kenderaan).

## 🚗 Overview

This package contains a complete end-to-end solution for processing Malaysian vehicle ownership certificates, including:

- **Layout-aware field extraction** using spatial analysis
- **Template-based extraction** with confidence scoring
- **Synthetic training dataset** with 1,000 annotated samples
- **Production API** components for integration
- **Comprehensive validation** and quality assurance
- **Multi-format export** capabilities

## 📁 Package Structure

```
vehicle_certificate_extraction_system/
├── core_extractors/                    # Main extraction engines
│   ├── layout_aware_field_extractor.py    # Primary vehicle certificate extractor
│   └── malaysian_vehicle_synthetic_generator.py  # Synthetic data generator
├── pipeline_components/                # Processing pipeline components
│   ├── complete_document_pipeline.py      # End-to-end processing pipeline
│   ├── field_extractor.py                 # Generic field extraction service
│   └── information_extraction.py          # Regex patterns and templates
├── synthetic_dataset/                  # Training and testing data
│   ├── dataset_summary.json              # Dataset statistics and metadata
│   ├── test/                             # Testing data (100 samples)
│   └── val/                              # Validation data (100 samples)
├── backend_integration/                # Backend database models and routes
│   ├── Vehicle.js                        # Database model for vehicles
│   └── vehicles.js                       # API routes for vehicle operations
├── api_components/                     # Production API components
│   ├── production_api_server.py          # FastAPI production server
│   └── integration_demo.py               # Integration demonstration
├── documentation/                      # Comprehensive documentation
│   ├── COMPLETE_PIPELINE_DOCUMENTATION.md
│   ├── USAGE_EXAMPLES.md
│   └── README.md
└── README.md                          # This file
```

## 🎯 Extractable Fields

The system can extract the following 12 fields from vehicle certificates:

| Field | Malay Name | Description |
|-------|------------|-------------|
| `plate_number` | No. Pendaftaran | Vehicle registration number |
| `owner_name` | Nama Pemilik | Owner's full name |
| `nric` | No. Kad Pengenalan | Owner's IC number |
| `address` | Alamat | Owner's address |
| `make` | Jenama | Vehicle manufacturer |
| `model` | Model | Vehicle model |
| `year` | Tahun Dibuat | Year of manufacture |
| `engine_number` | No. Enjin | Engine number |
| `chassis_number` | No. Casis | Chassis number |
| `color` | Warna | Vehicle color |
| `fuel_type` | Jenis Bahan Api | Fuel type |
| `engine_capacity` | Isi Padu | Engine capacity |

## 🚀 Quick Start

### 1. Basic Field Extraction

```python
from core_extractors.layout_aware_field_extractor import TemplateBasedExtractor

# Initialize extractor
extractor = TemplateBasedExtractor()

# Load OCR results (from your OCR engine)
with open('vehicle_ocr_results.json', 'r') as f:
    ocr_results = json.load(f)

# Extract fields
extracted_fields = extractor.extract_fields(ocr_results)

# Access extracted data
for field in extracted_fields:
    print(f"{field.field_name}: {field.value_region.text} (confidence: {field.confidence})")
```

### 2. Complete Pipeline Processing

```python
from pipeline_components.complete_document_pipeline import MalaysianVehicleProcessor

# Initialize processor
processor = MalaysianVehicleProcessor()

# Process vehicle certificate image
results = processor.process_document('vehicle_certificate.jpg')

# Get extracted fields
extracted_fields = results['extracted_fields']
validation_results = results['validation_results']
```

### 3. Command Line Usage

```bash
# Extract fields from OCR results
python core_extractors/layout_aware_field_extractor.py vehicle_ocr.json --output extracted_fields.json

# Run complete pipeline
python pipeline_components/complete_document_pipeline.py vehicle_image.jpg --output results/
```

## 📊 Dataset Information

- **Total Samples**: 1,000 annotated vehicle certificates
- **Training Set**: 800 samples
- **Validation Set**: 100 samples  
- **Test Set**: 100 samples
- **Image Size**: 800x1200 pixels
- **Format**: JPG images with JSON annotations

## 🔧 System Requirements

### Python Dependencies
```
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.3.0
paddleocr>=2.6.0  # or pytesseract>=0.3.8
fastapi>=0.68.0
uvicorn>=0.15.0
```

### Optional Dependencies
```
torch>=1.9.0  # For advanced ML features
transformers>=4.11.0  # For NLP-based extraction
scikit-learn>=1.0.0  # For classification
```

## 🏗️ Architecture

### Core Components

1. **Layout-Aware Extractor**: Uses spatial analysis and template matching
2. **Context Analyzer**: Understands field relationships and positioning
3. **Validation Engine**: Ensures extracted data quality and consistency
4. **Quality Assurance**: Automated review and confidence scoring

### Processing Pipeline

1. **Image Preprocessing**: Noise reduction, deskewing, enhancement
2. **OCR Processing**: Text extraction with bounding boxes
3. **Field Extraction**: Template-based and spatial analysis
4. **Validation**: Format checking and cross-field validation
5. **Export**: Multiple format support (JSON, CSV, XML)

## 📈 Performance Metrics

- **Average Processing Time**: ~9.39 seconds per document
- **Field Extraction Accuracy**: 95%+ for most fields
- **Validation Success Rate**: 95%+
- **Confidence Scoring**: Available for all extracted fields

## 🔌 API Integration

### FastAPI Endpoints

```python
# Process document
POST /process-document
Content-Type: multipart/form-data

# Get processing status
GET /status/{job_id}

# Retrieve results
GET /results/{job_id}
```

### Database Integration

The system includes Sequelize models for storing extracted vehicle data:

```javascript
// Vehicle model with validation
const vehicle = await Vehicle.create({
  plateNumber: extracted_fields.plate_number,
  ownerName: extracted_fields.owner_name,
  nric: extracted_fields.nric,
  // ... other fields
});
```

## 🛡️ Validation Features

- **Format Validation**: IC numbers, plate numbers, dates
- **Range Validation**: Engine capacity, year constraints
- **Cross-Field Validation**: Consistency checks between related fields
- **Confidence Thresholds**: Configurable quality gates

## 📝 Usage Examples

See `documentation/USAGE_EXAMPLES.md` for comprehensive examples including:

- Basic field extraction
- Batch processing
- Custom validation rules
- API integration
- Error handling

## 🔍 Troubleshooting

### Common Issues

1. **Low OCR Quality**: Ensure good image preprocessing
2. **Missing Fields**: Check template configuration
3. **Validation Failures**: Review field format requirements
4. **Performance Issues**: Consider image resizing for large files

### Debug Mode

```python
# Enable verbose logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Use debug extraction
extractor = TemplateBasedExtractor(debug=True)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or contributions:

1. Check the documentation in `documentation/`
2. Review usage examples
3. Create an issue with detailed information
4. Include sample images and OCR results for debugging

## 🔄 Version History

- **v1.0.0**: Initial release with complete extraction system
- **v1.1.0**: Added synthetic dataset and training pipeline
- **v1.2.0**: Production API and quality assurance features

---

**Note**: This system is specifically designed for Malaysian Vehicle Registration Certificates. For other document types, see the main project documentation.