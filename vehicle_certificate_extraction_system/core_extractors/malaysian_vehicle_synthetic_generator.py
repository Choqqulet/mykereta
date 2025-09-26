#!/usr/bin/env python3
"""
Malaysian Vehicle Registration Certificate Synthetic Data Generator
Generates thousands of realistic variants with proper Malaysian data elements and visual variations.
"""

import os
import json
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
from dataclasses import dataclass
import logging
from pathlib import Path
import albumentations as A

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation"""
    num_samples: int = 5000
    output_dir: str = "synthetic_dataset"
    image_size: Tuple[int, int] = (800, 1200)  # width, height
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

class MalaysianDataGenerator:
    """Generates realistic Malaysian data elements"""
    
    def __init__(self):
        self.malay_names = [
            "AHMAD", "ALI", "HASSAN", "IBRAHIM", "ISMAIL", "MOHAMED", "OMAR", "RAHMAN", "SALLEH", "YUSOF",
            "FATIMAH", "KHADIJAH", "MARYAM", "NOOR", "SITI", "ZAINAB", "AISHAH", "HALIMAH", "ROHANI", "ZALEHA"
        ]
        
        self.chinese_names = [
            "TAN", "LIM", "LEE", "ONG", "WONG", "CHAN", "GOH", "TEO", "NG", "CHONG",
            "WEI", "MING", "LING", "YEN", "HUI", "MEI", "LI", "XIAO", "JUN", "FENG"
        ]
        
        self.indian_names = [
            "KUMAR", "SINGH", "DEVI", "SHARMA", "PATEL", "REDDY", "RAO", "KRISHNAN", "MURUGAN", "RAMAN",
            "PRIYA", "KAVITHA", "MEERA", "RADHA", "LAKSHMI", "GEETHA", "SHANTI", "KAMALA", "VANI", "MAYA"
        ]
        
        self.honorifics = ["", "EN.", "PN.", "DATO'", "DATUK", "TAN SRI", "PROF.", "DR."]
        
        self.malaysian_states = [
            "JOHOR", "KEDAH", "KELANTAN", "MELAKA", "NEGERI SEMBILAN", "PAHANG", 
            "PERAK", "PERLIS", "PULAU PINANG", "SABAH", "SARAWAK", "SELANGOR", 
            "TERENGGANU", "KUALA LUMPUR", "LABUAN", "PUTRAJAYA"
        ]
        
        self.vehicle_makes = [
            "PROTON", "PERODUA", "TOYOTA", "HONDA", "NISSAN", "MAZDA", "MITSUBISHI",
            "HYUNDAI", "KIA", "VOLKSWAGEN", "BMW", "MERCEDES-BENZ", "AUDI", "FORD"
        ]
        
        self.vehicle_models = {
            "PROTON": ["SAGA", "PERSONA", "IRIZ", "EXORA", "X70", "X50"],
            "PERODUA": ["MYVI", "AXIA", "BEZZA", "ALZA", "ARUZ", "ATIVA"],
            "TOYOTA": ["VIOS", "CAMRY", "COROLLA", "INNOVA", "FORTUNER", "HILUX"],
            "HONDA": ["CITY", "CIVIC", "ACCORD", "CR-V", "HR-V", "BR-V"],
            "NISSAN": ["ALMERA", "TEANA", "X-TRAIL", "NAVARA", "LIVINA", "SERENA"]
        }
        
        self.colors = [
            "PUTIH", "HITAM", "KELABU", "PERAK", "MERAH", "BIRU", "HIJAU", 
            "KUNING", "OREN", "UNGU", "COKLAT", "EMAS"
        ]
        
        self.fuel_types = ["PETROL", "DIESEL", "HYBRID", "ELEKTRIK"]
        
        self.street_prefixes = [
            "JALAN", "LORONG", "TAMAN", "KAMPUNG", "BANDAR", "PERSIARAN", 
            "LEBUH", "JALAN RAJA", "JALAN TUN", "JALAN DATO'"
        ]
        
        self.area_names = [
            "BUKIT BINTANG", "CHERAS", "PETALING JAYA", "SHAH ALAM", "SUBANG JAYA",
            "AMPANG", "KAJANG", "KLANG", "SEREMBAN", "JOHOR BAHRU", "IPOH", "PENANG"
        ]
    
    def generate_name(self) -> str:
        """Generate a realistic Malaysian name"""
        ethnicity = random.choice(["malay", "chinese", "indian"])
        honorific = random.choice(self.honorifics)
        
        if ethnicity == "malay":
            first_name = random.choice(self.malay_names)
            if random.random() > 0.3:  # 70% chance of having "BIN" or "BINTI"
                connector = "BIN" if random.random() > 0.5 else "BINTI"
                last_name = random.choice(self.malay_names)
                full_name = f"{first_name} {connector} {last_name}"
            else:
                full_name = first_name
        elif ethnicity == "chinese":
            first_name = random.choice(self.chinese_names)
            last_name = random.choice(self.chinese_names)
            full_name = f"{first_name} {last_name}"
        else:  # indian
            first_name = random.choice(self.indian_names)
            if random.random() > 0.4:  # 60% chance of having "A/L" or "A/P"
                connector = "A/L" if random.random() > 0.5 else "A/P"
                last_name = random.choice(self.indian_names)
                full_name = f"{first_name} {connector} {last_name}"
            else:
                full_name = first_name
        
        if honorific:
            return f"{honorific} {full_name}"
        return full_name
    
    def generate_nric(self) -> str:
        """Generate a valid-looking Malaysian NRIC number"""
        # Generate birth date (1950-2005)
        start_date = datetime(1950, 1, 1)
        end_date = datetime(2005, 12, 31)
        birth_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        
        # Format: YYMMDD-PB-####
        year = birth_date.strftime("%y")
        month = birth_date.strftime("%m")
        day = birth_date.strftime("%d")
        
        # Place of birth (state codes)
        pb_codes = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
        pb = random.choice(pb_codes)
        
        # Last 4 digits
        last_four = f"{random.randint(0, 9999):04d}"
        
        return f"{year}{month}{day}-{pb}-{last_four}"
    
    def generate_plate_number(self) -> str:
        """Generate a realistic Malaysian vehicle plate number"""
        # Common Malaysian plate formats
        formats = [
            "W{}{} {}",  # Wilayah format
            "B{}{} {}",  # Selangor format
            "A{}{} {}",  # Perak format
            "J{}{} {}",  # Johor format
            "P{}{} {}",  # Penang format
            "K{}{} {}",  # Kedah format
        ]
        
        format_choice = random.choice(formats)
        
        # Generate numbers and letters
        letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        numbers = f"{random.randint(1, 9999):04d}"
        
        return format_choice.format(letters, numbers[:2], numbers[2:])
    
    def generate_vin(self) -> str:
        """Generate a realistic VIN number"""
        # VIN format: 17 characters, excluding I, O, Q
        allowed_chars = string.ascii_uppercase.replace('I', '').replace('O', '').replace('Q', '') + string.digits
        return ''.join(random.choices(allowed_chars, k=17))
    
    def generate_engine_number(self) -> str:
        """Generate an engine number"""
        prefix = ''.join(random.choices(string.ascii_uppercase, k=2))
        numbers = ''.join(random.choices(string.digits, k=6))
        return f"{prefix}{numbers}"
    
    def generate_address(self) -> str:
        """Generate a realistic Malaysian address"""
        # House number
        house_num = random.randint(1, 999)
        
        # Street
        street_prefix = random.choice(self.street_prefixes)
        street_name = random.choice(self.area_names)
        
        # Area
        area = random.choice(self.area_names)
        
        # Postcode (5 digits)
        postcode = f"{random.randint(10000, 99999)}"
        
        # State
        state = random.choice(self.malaysian_states)
        
        address_lines = [
            f"{house_num}, {street_prefix} {street_name}",
            f"{area}",
            f"{postcode} {state}"
        ]
        
        return "\n".join(address_lines)
    
    def generate_vehicle_data(self) -> Dict[str, str]:
        """Generate complete vehicle data"""
        make = random.choice(self.vehicle_makes)
        model = random.choice(self.vehicle_models.get(make, ["UNKNOWN"]))
        year = str(random.randint(2000, 2023))
        color = random.choice(self.colors)
        fuel_type = random.choice(self.fuel_types)
        engine_capacity = f"{random.randint(1000, 3500)}CC"
        
        return {
            "No. Pendaftaran": self.generate_plate_number(),
            "Nama Pemilik": self.generate_name(),
            "No. Kad Pengenalan": self.generate_nric(),
            "Alamat": self.generate_address(),
            "Jenama": make,
            "Model": model,
            "Tahun Dibuat": year,
            "No. Enjin": self.generate_engine_number(),
            "No. Casis": self.generate_vin(),
            "Warna": color,
            "Jenis Bahan Api": fuel_type,
            "Isi Padu": engine_capacity
        }

class DocumentRenderer:
    """Renders synthetic documents with realistic visual variations"""
    
    def __init__(self, template_path: str):
        self.template = self.load_template(template_path)
        self.fonts = self.load_fonts()
        self.augmentation_pipeline = self.create_augmentation_pipeline()
    
    def load_template(self, template_path: str):
        """Load document template"""
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_fonts(self) -> Dict[str, List[ImageFont.FreeTypeFont]]:
        """Load various fonts for text rendering"""
        fonts = {
            'label': [],
            'value': [],
            'header': []
        }
        
        # Try to load system fonts
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Times.ttc",
            "/System/Library/Fonts/Courier.ttc"
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    fonts['label'].append(ImageFont.truetype(font_path, 12))
                    fonts['value'].append(ImageFont.truetype(font_path, 14))
                    fonts['header'].append(ImageFont.truetype(font_path, 16))
                except:
                    pass
        
        # Fallback to default font if no fonts loaded
        if not fonts['label']:
            fonts['label'] = [ImageFont.load_default()]
            fonts['value'] = [ImageFont.load_default()]
            fonts['header'] = [ImageFont.load_default()]
        
        return fonts
    
    def create_augmentation_pipeline(self):
        """Create augmentation pipeline for visual variations"""
        return A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.Rotate(limit=5, p=0.3),
            A.Perspective(scale=(0.05, 0.1), p=0.2),
        ])
    
    def render_document(self, data: Dict[str, str]) -> Tuple[Image.Image, Dict]:
        """Render a synthetic document with the given data"""
        # Create base image
        width, height = self.template['image_size']
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Add subtle background texture
        self.add_background_texture(image)
        
        # Render fields
        annotations = []
        for field_info in self.template['fields']:
            field_label = field_info['label']
            bbox = field_info['bbox']
            field_type = field_info.get('field_type', 'text')
            is_multiline = field_info.get('is_multiline', False)
            
            if field_label in data:
                text = data[field_label]
                
                # Render the field
                actual_bbox = self.render_field(
                    draw, text, bbox, field_type, is_multiline
                )
                
                # Store annotation
                annotations.append({
                    'label': field_label,
                    'bbox': actual_bbox,
                    'text': text,
                    'field_type': field_type
                })
        
        # Add document header/title
        self.add_document_header(draw, width)
        
        # Add stamps and signatures
        self.add_stamps_and_signatures(image)
        
        # Apply augmentations
        image_array = np.array(image)
        augmented = self.augmentation_pipeline(image=image_array)['image']
        final_image = Image.fromarray(augmented)
        
        return final_image, {'fields': annotations}
    
    def render_field(self, draw: ImageDraw.Draw, text: str, bbox: List[int], 
                    field_type: str, is_multiline: bool) -> List[int]:
        """Render a single field with appropriate formatting"""
        x, y, w, h = bbox
        
        # Choose font based on field type
        if field_type in ['header', 'title']:
            font = random.choice(self.fonts['header'])
        elif field_type in ['label']:
            font = random.choice(self.fonts['label'])
        else:
            font = random.choice(self.fonts['value'])
        
        # Add some positional jitter
        jitter_x = random.randint(-5, 5)
        jitter_y = random.randint(-3, 3)
        x += jitter_x
        y += jitter_y
        
        # Text color with slight variation
        base_color = (0, 0, 0)
        color_variation = random.randint(-30, 30)
        text_color = tuple(max(0, min(255, c + color_variation)) for c in base_color)
        
        if is_multiline and '\n' in text:
            # Handle multiline text
            lines = text.split('\n')
            bbox = draw.textbbox((0, 0), 'A', font=font)
            line_height = bbox[3] - bbox[1] + 2
            for i, line in enumerate(lines):
                draw.text((x, y + i * line_height), line, font=font, fill=text_color)
            actual_height = len(lines) * line_height
        else:
            # Single line text
            draw.text((x, y), text, font=font, fill=text_color)
            bbox = draw.textbbox((0, 0), text, font=font)
            actual_height = bbox[3] - bbox[1]
        
        # Calculate actual text width
        if is_multiline and '\n' in text:
            actual_width = max(draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in text.split('\n'))
        else:
            bbox = draw.textbbox((0, 0), text, font=font)
            actual_width = bbox[2] - bbox[0]
        
        return [x, y, actual_width, actual_height]
    
    def add_background_texture(self, image: Image.Image):
        """Add subtle background texture to make document look more realistic"""
        # Create a subtle paper texture
        width, height = image.size
        
        # Generate noise pattern
        noise = np.random.normal(245, 10, (height, width, 3))
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        
        # Create texture image
        texture = Image.fromarray(noise, 'RGB')
        
        # Blend with white background
        image.paste(texture, (0, 0))
    
    def add_document_header(self, draw: ImageDraw.Draw, width: int):
        """Add document header/title"""
        header_text = "SIJIL PEMILIKAN KENDERAAN"
        font = random.choice(self.fonts['header'])
        
        # Center the header
        bbox = draw.textbbox((0, 0), header_text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = 20
        
        draw.text((x, y), header_text, font=font, fill=(0, 0, 0))
    
    def add_stamps_and_signatures(self, image: Image.Image):
        """Add realistic stamps and signature overlays"""
        # Add a simple circular stamp
        if random.random() > 0.7:  # 30% chance
            self.add_circular_stamp(image)
        
        # Add signature-like scribble
        if random.random() > 0.6:  # 40% chance
            self.add_signature(image)
    
    def add_circular_stamp(self, image: Image.Image):
        """Add a circular stamp overlay"""
        draw = ImageDraw.Draw(image)
        
        # Random position
        x = random.randint(50, image.width - 100)
        y = random.randint(image.height // 2, image.height - 100)
        radius = random.randint(30, 50)
        
        # Draw circle
        color = (random.randint(0, 100), random.randint(0, 100), random.randint(150, 255))
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, width=2)
        
        # Add stamp text
        stamp_text = "LULUS"
        font = random.choice(self.fonts['label'])
        bbox = draw.textbbox((0, 0), stamp_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width//2, y - text_height//2), stamp_text, font=font, fill=color)
    
    def add_signature(self, image: Image.Image):
        """Add a signature-like scribble"""
        draw = ImageDraw.Draw(image)
        
        # Random position in lower part of document
        start_x = random.randint(50, image.width - 200)
        start_y = random.randint(image.height - 200, image.height - 50)
        
        # Generate signature-like path
        points = [(start_x, start_y)]
        for i in range(10):
            prev_x, prev_y = points[-1]
            new_x = prev_x + random.randint(-20, 20)
            new_y = prev_y + random.randint(-10, 10)
            points.append((new_x, new_y))
        
        # Draw signature
        color = (random.randint(0, 50), random.randint(0, 50), random.randint(100, 200))
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=color, width=2)

class SyntheticDatasetGenerator:
    """Main class for generating synthetic datasets"""
    
    def __init__(self, config: SyntheticDataConfig, template_path: str):
        self.config = config
        self.data_generator = MalaysianDataGenerator()
        self.renderer = DocumentRenderer(template_path)
        
        # Create output directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for dataset"""
        base_dir = Path(self.config.output_dir)
        base_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (base_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (base_dir / split / 'annotations').mkdir(parents=True, exist_ok=True)
    
    def generate_dataset(self):
        """Generate the complete synthetic dataset"""
        logger.info(f"Generating {self.config.num_samples} synthetic samples...")
        
        # Calculate split sizes
        train_size = int(self.config.num_samples * self.config.train_split)
        val_size = int(self.config.num_samples * self.config.val_split)
        test_size = self.config.num_samples - train_size - val_size
        
        splits = [
            ('train', train_size),
            ('val', val_size),
            ('test', test_size)
        ]
        
        sample_id = 0
        for split_name, split_size in splits:
            logger.info(f"Generating {split_size} samples for {split_name} split...")
            
            for i in range(split_size):
                # Generate data
                vehicle_data = self.data_generator.generate_vehicle_data()
                
                # Render document
                image, annotations = self.renderer.render_document(vehicle_data)
                
                # Save image
                image_filename = f"sample_{sample_id:06d}.jpg"
                image_path = Path(self.config.output_dir) / split_name / 'images' / image_filename
                image.save(image_path, 'JPEG', quality=random.randint(85, 95))
                
                # Save annotations
                annotation_filename = f"sample_{sample_id:06d}.json"
                annotation_path = Path(self.config.output_dir) / split_name / 'annotations' / annotation_filename
                
                annotation_data = {
                    'image_filename': image_filename,
                    'image_size': [image.width, image.height],
                    'vehicle_data': vehicle_data,
                    'annotations': annotations
                }
                
                with open(annotation_path, 'w', encoding='utf-8') as f:
                    json.dump(annotation_data, f, indent=2, ensure_ascii=False)
                
                sample_id += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i + 1}/{split_size} samples for {split_name}")
        
        # Generate dataset summary
        self.generate_dataset_summary()
        
        logger.info(f"Dataset generation completed! Output directory: {self.config.output_dir}")
    
    def generate_dataset_summary(self):
        """Generate a summary of the dataset"""
        summary = {
            'total_samples': self.config.num_samples,
            'splits': {
                'train': int(self.config.num_samples * self.config.train_split),
                'val': int(self.config.num_samples * self.config.val_split),
                'test': int(self.config.num_samples * self.config.test_split)
            },
            'image_size': self.config.image_size,
            'field_types': [
                'plate_number', 'owner_name', 'nric', 'address', 'make', 'model',
                'year', 'engine_number', 'chassis_number', 'color', 'fuel_type', 'engine_capacity'
            ],
            'generation_date': datetime.now().isoformat()
        }
        
        summary_path = Path(self.config.output_dir) / 'dataset_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

def main():
    """Main function to generate synthetic dataset"""
    # Configuration
    config = SyntheticDataConfig(
        num_samples=1000,  # Start with 1000 samples for testing
        output_dir="synthetic_vehicle_dataset",
        image_size=(800, 1200)
    )
    
    # Template path (will be created by document analysis)
    template_path = "templates/vehicle_registration_manual_template.json"
    
    if not os.path.exists(template_path):
        logger.error(f"Template not found: {template_path}")
        logger.error("Please run document_analysis_pipeline.py first to create the template")
        return
    
    try:
        # Generate dataset
        generator = SyntheticDatasetGenerator(config, template_path)
        generator.generate_dataset()
        
        logger.info("Synthetic dataset generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        raise

if __name__ == "__main__":
    main()