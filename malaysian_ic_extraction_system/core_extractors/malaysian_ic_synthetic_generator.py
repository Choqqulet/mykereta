#!/usr/bin/env python3
"""
Malaysian IC (MyKad) Synthetic Data Generator

Generates realistic synthetic Malaysian Identity Cards with:
- Valid NRIC numbers (YYMMDD-SS-NNNN format)
- Authentic Malaysian names (Malay, Chinese, Indian)
- Real Malaysian addresses with postal codes
- Proper field validation and consistency
- Visual variations for training data augmentation

Author: AI Assistant
Date: 2025
"""

import random
import json
import os
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
from typing import Dict, List, Tuple, Optional
import uuid
import argparse

class MalaysianNameGenerator:
    """Generates authentic Malaysian names across different ethnicities"""
    
    def __init__(self):
        # Malay names
        self.malay_male_names = [
            "AHMAD", "MUHAMMAD", "ALI", "HASSAN", "IBRAHIM", "ISMAIL", "OMAR", "YUSOF",
            "ABDUL RAHMAN", "ABDUL AZIZ", "MOHD", "FARID", "AZMAN", "RAZAK", "ZULKIFLI",
            "ROSLI", "KAMAL", "SULAIMAN", "ZAKARIA", "HAKIM", "FADZIL", "NAZIR"
        ]
        
        self.malay_female_names = [
            "SITI", "NUR", "FATIMAH", "AISHAH", "KHADIJAH", "ZAINAB", "ROHANI", "NORAINI",
            "FARIDAH", "AMINAH", "RAMLAH", "HALIMAH", "MARIAM", "ZALEHA", "ROSNAH",
            "NORMAH", "SALIMAH", "RASHIDAH", "HASNAH", "RUSNAH", "ZAHARAH", "NORHAYATI"
        ]
        
        self.malay_surnames = [
            "BIN AHMAD", "BIN IBRAHIM", "BIN HASSAN", "BIN ALI", "BIN OMAR", "BIN YUSOF",
            "BINTI AHMAD", "BINTI IBRAHIM", "BINTI HASSAN", "BINTI ALI", "BINTI OMAR",
            "BIN ABDUL RAHMAN", "BIN MOHD", "BIN SULAIMAN", "BINTI ABDUL RAHMAN"
        ]
        
        # Chinese names
        self.chinese_surnames = ["TAN", "LIM", "LEE", "ONG", "WONG", "CHAN", "CHONG", "GOH", "TEO", "YAP"]
        self.chinese_given_names = [
            "WEI MING", "JIA WEI", "YI CHEN", "ZI YANG", "HAO RAN", "JUN HAO", "YU XUAN",
            "MEI LING", "LI YING", "XIN YI", "JIA YI", "YU TING", "WEI LING", "SHU HUI"
        ]
        
        # Indian names
        self.indian_male_names = [
            "RAMAN", "KUMAR", "SURESH", "RAJESH", "PRAKASH", "DEEPAK", "ANIL", "VIJAY",
            "SANJAY", "ASHOK", "MOHAN", "GANESH", "KRISHNAN", "BALAN", "SELVAM"
        ]
        
        self.indian_female_names = [
            "PRIYA", "KAVITHA", "MEERA", "SITA", "GEETHA", "RADHA", "KAMALA", "SHANTI",
            "DEVI", "LAKSHMI", "SARASWATI", "INDIRA", "MALATHI", "VASANTHA", "PREMA"
        ]
        
        self.indian_surnames = [
            "A/L RAMAN", "A/L KUMAR", "A/L SURESH", "A/L PRAKASH", "A/L MOHAN",
            "A/P RAMAN", "A/P KUMAR", "A/P SURESH", "A/P PRAKASH", "A/P MOHAN"
        ]
    
    def generate_name(self, gender: str, ethnicity: str = None) -> str:
        """Generate a name based on gender and ethnicity"""
        if ethnicity is None:
            ethnicity = random.choice(["malay", "chinese", "indian"])
        
        if ethnicity == "malay":
            if gender == "LELAKI":
                first_name = random.choice(self.malay_male_names)
                surname = random.choice([s for s in self.malay_surnames if "BIN" in s])
            else:
                first_name = random.choice(self.malay_female_names)
                surname = random.choice([s for s in self.malay_surnames if "BINTI" in s])
            return f"{first_name} {surname}"
        
        elif ethnicity == "chinese":
            surname = random.choice(self.chinese_surnames)
            given_name = random.choice(self.chinese_given_names)
            return f"{surname} {given_name}"
        
        elif ethnicity == "indian":
            if gender == "LELAKI":
                first_name = random.choice(self.indian_male_names)
                surname = random.choice([s for s in self.indian_surnames if "A/L" in s])
            else:
                first_name = random.choice(self.indian_female_names)
                surname = random.choice([s for s in self.indian_surnames if "A/P" in s])
            return f"{first_name} {surname}"
        
        return "UNKNOWN NAME"

class MalaysianAddressGenerator:
    """Generates realistic Malaysian addresses with proper postal codes"""
    
    def __init__(self):
        self.states = {
            "JOHOR": ["80000", "81000", "82000", "83000", "84000", "85000", "86000"],
            "KEDAH": ["05000", "06000", "07000", "08000", "09000"],
            "KELANTAN": ["15000", "16000", "17000", "18000"],
            "MELAKA": ["75000", "76000", "77000", "78000"],
            "NEGERI SEMBILAN": ["70000", "71000", "72000", "73000"],
            "PAHANG": ["25000", "26000", "27000", "28000", "39000"],
            "PERAK": ["30000", "31000", "32000", "33000", "34000", "35000", "36000"],
            "PERLIS": ["01000", "02000"],
            "PULAU PINANG": ["10000", "11000", "12000", "13000", "14000"],
            "SABAH": ["88000", "89000", "90000", "91000"],
            "SARAWAK": ["93000", "94000", "95000", "96000", "97000", "98000"],
            "SELANGOR": ["40000", "41000", "42000", "43000", "44000", "45000", "46000", "47000", "48000"],
            "TERENGGANU": ["20000", "21000", "22000", "23000", "24000"],
            "KUALA LUMPUR": ["50000", "51000", "52000", "53000", "54000", "55000", "56000", "57000", "58000", "59000"],
            "LABUAN": ["87000"],
            "PUTRAJAYA": ["62000", "62050", "62100", "62150", "62200", "62250", "62300"]
        }
        
        self.street_types = ["JALAN", "LORONG", "PERSIARAN", "LEBUH", "TINGKAT"]
        self.street_names = [
            "BUNGA RAYA", "MERDEKA", "SULTAN", "RAJA", "DATUK", "TUN", "BANGSAR",
            "AMPANG", "CHERAS", "SETAPAK", "WANGSA MAJU", "TAMAN", "BANDAR",
            "SERI", "INDAH", "CEMERLANG", "HARMONI", "SEJAHTERA", "BAHAGIA"
        ]
        
        self.building_types = ["NO", "LOT", "TINGKAT", "BLOK", "UNIT"]
    
    def generate_address(self) -> str:
        """Generate a realistic Malaysian address"""
        # Building number/unit
        building_type = random.choice(self.building_types)
        building_num = random.randint(1, 999)
        
        # Street
        street_type = random.choice(self.street_types)
        street_name = random.choice(self.street_names)
        
        # State and postal code
        state = random.choice(list(self.states.keys()))
        postal_code = random.choice(self.states[state])
        
        # Optional area/district
        areas = ["TAMAN", "BANDAR", "KAMPUNG", "DESA"]
        area = f"{random.choice(areas)} {random.choice(self.street_names)}"
        
        address_parts = [
            f"{building_type} {building_num}",
            f"{street_type} {street_name}",
            area,
            f"{postal_code} {state}"
        ]
        
        return ", ".join(address_parts)

class NRICGenerator:
    """Generates valid Malaysian NRIC numbers with proper validation"""
    
    def __init__(self):
        # Malaysian state codes
        self.state_codes = {
            "01": "JOHOR", "02": "KEDAH", "03": "KELANTAN", "04": "MELAKA",
            "05": "NEGERI SEMBILAN", "06": "PAHANG", "07": "PERAK", "08": "PERLIS",
            "09": "PULAU PINANG", "10": "SELANGOR", "11": "TERENGGANU", "12": "SABAH",
            "13": "SARAWAK", "14": "KUALA LUMPUR", "15": "LABUAN", "16": "PUTRAJAYA",
            "21": "JOHOR", "22": "JOHOR", "23": "JOHOR", "24": "JOHOR",
            "25": "KEDAH", "26": "KEDAH", "27": "KEDAH",
            "28": "KELANTAN", "29": "KELANTAN",
            "30": "MELAKA",
            "31": "NEGERI SEMBILAN", "32": "NEGERI SEMBILAN",
            "33": "PAHANG", "34": "PAHANG", "35": "PAHANG", "36": "PAHANG",
            "37": "PERAK", "38": "PERAK", "39": "PERAK", "40": "PERAK",
            "41": "PERLIS",
            "42": "PULAU PINANG", "43": "PULAU PINANG", "44": "PULAU PINANG",
            "45": "SELANGOR", "46": "SELANGOR", "47": "SELANGOR", "48": "SELANGOR",
            "49": "TERENGGANU", "50": "TERENGGANU",
            "51": "SABAH", "52": "SABAH", "53": "SABAH", "54": "SABAH", "55": "SABAH",
            "56": "SARAWAK", "57": "SARAWAK", "58": "SARAWAK", "59": "SARAWAK",
            "60": "KUALA LUMPUR",
            "61": "LABUAN",
            "62": "PUTRAJAYA",
            "71": "FOREIGN BORN", "72": "FOREIGN BORN", "74": "FOREIGN BORN",
            "75": "FOREIGN BORN", "76": "FOREIGN BORN", "77": "FOREIGN BORN",
            "78": "FOREIGN BORN", "79": "FOREIGN BORN"
        }
    
    def generate_nric(self, birth_date: datetime, gender: str) -> str:
        """Generate valid NRIC based on birth date and gender"""
        # Format: YYMMDD-SS-NNNN
        year = birth_date.strftime("%y")
        month = birth_date.strftime("%m")
        day = birth_date.strftime("%d")
        
        # State code (random)
        state_code = random.choice(list(self.state_codes.keys()))
        
        # Last 4 digits: odd for male, even for female
        if gender == "LELAKI":
            last_digit = random.choice([1, 3, 5, 7, 9])
        else:
            last_digit = random.choice([0, 2, 4, 6, 8])
        
        first_three = random.randint(100, 999)
        last_four = f"{first_three}{last_digit}"
        
        return f"{year}{month}{day}-{state_code}-{last_four}"
    
    def validate_nric(self, nric: str) -> bool:
        """Validate NRIC format and consistency"""
        import re
        pattern = r"^\d{6}-\d{2}-\d{4}$"
        return bool(re.match(pattern, nric))

class MalaysianICGenerator:
    """Main class for generating synthetic Malaysian IC data"""
    
    def __init__(self):
        self.name_generator = MalaysianNameGenerator()
        self.address_generator = MalaysianAddressGenerator()
        self.nric_generator = NRICGenerator()
        
        self.religions = ["ISLAM", "BUDDHA", "HINDU", "KRISTIAN", "TAOISME", "KONFUSIANISME", "LAIN-LAIN"]
        self.nationalities = ["WARGANEGARA", "BUKAN WARGANEGARA"]
        
        # Font paths (you may need to adjust these)
        self.font_paths = {
            "regular": "/System/Library/Fonts/Arial.ttf",
            "bold": "/System/Library/Fonts/Arial Bold.ttf"
        }
    
    def generate_birth_date(self) -> datetime:
        """Generate random birth date between 1950 and 2005"""
        start_date = datetime(1950, 1, 1)
        end_date = datetime(2005, 12, 31)
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        
        return start_date + timedelta(days=random_days)
    
    def generate_issue_date(self, birth_date: datetime) -> datetime:
        """Generate card issue date (must be after 18th birthday)"""
        min_issue = birth_date + timedelta(days=18*365)
        max_issue = datetime.now()
        
        if min_issue > max_issue:
            min_issue = max_issue - timedelta(days=365)
        
        time_between = max_issue - min_issue
        days_between = time_between.days
        random_days = random.randrange(max(1, days_between))
        
        return min_issue + timedelta(days=random_days)
    
    def generate_ic_data(self) -> Dict:
        """Generate complete IC data"""
        # Basic demographics
        gender = random.choice(["LELAKI", "PEREMPUAN"])
        birth_date = self.generate_birth_date()
        issue_date = self.generate_issue_date(birth_date)
        
        # Generate NRIC
        nric = self.nric_generator.generate_nric(birth_date, gender)
        
        # Generate name (ethnicity affects religion probability)
        ethnicity = random.choice(["malay", "chinese", "indian"])
        name = self.name_generator.generate_name(gender, ethnicity)
        
        # Religion based on ethnicity (realistic distribution)
        if ethnicity == "malay":
            religion = "ISLAM"
        elif ethnicity == "chinese":
            religion = random.choice(["BUDDHA", "TAOISME", "KRISTIAN"])
        else:  # indian
            religion = random.choice(["HINDU", "KRISTIAN", "ISLAM"])
        
        # Address and nationality
        address = self.address_generator.generate_address()
        nationality = random.choice(self.nationalities)
        
        return {
            "name": name,
            "nric": nric,
            "gender": gender,
            "birth_date": birth_date.strftime("%d-%m-%Y"),
            "address": address,
            "religion": religion,
            "nationality": nationality,
            "issue_date": issue_date.strftime("%d-%m-%Y"),
            "ethnicity": ethnicity
        }
    
    def create_ic_image(self, ic_data: Dict, output_path: str, 
                       width: int = 856, height: int = 540) -> str:
        """Create synthetic IC image with realistic layout"""
        
        # Create base image with gradient background
        img = Image.new('RGB', (width, height), color='#E8F4FD')
        draw = ImageDraw.Draw(img)
        
        # Add gradient background
        for y in range(height):
            color_value = int(232 + (244-232) * y / height)
            draw.line([(0, y), (width, y)], fill=(color_value, 244, 253))
        
        # Load fonts
        try:
            font_regular = ImageFont.truetype(self.font_paths["regular"], 24)
            font_bold = ImageFont.truetype(self.font_paths["bold"], 28)
            font_small = ImageFont.truetype(self.font_paths["regular"], 20)
            font_title = ImageFont.truetype(self.font_paths["bold"], 32)
        except:
            # Fallback to default font
            font_regular = ImageFont.load_default()
            font_bold = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_title = ImageFont.load_default()
        
        # Draw IC border
        border_color = '#2E5BBA'
        draw.rectangle([10, 10, width-10, height-10], outline=border_color, width=3)
        
        # Title
        draw.text((50, 30), "MALAYSIA", font=font_title, fill='#2E5BBA')
        draw.text((50, 70), "IDENTITY CARD / KAD PENGENALAN", font=font_bold, fill='#2E5BBA')
        
        # Photo placeholder
        photo_x, photo_y = 50, 120
        photo_w, photo_h = 120, 150
        draw.rectangle([photo_x, photo_y, photo_x + photo_w, photo_y + photo_h], 
                      fill='#CCCCCC', outline='#888888', width=2)
        draw.text((photo_x + 30, photo_y + 70), "PHOTO", font=font_small, fill='#666666')
        
        # Fields layout
        field_x = photo_x + photo_w + 30
        field_y = 120
        line_height = 35
        
        fields = [
            ("NAMA / NAME:", ic_data["name"]),
            ("NO. KAD PENGENALAN:", ic_data["nric"]),
            ("JANTINA / SEX:", ic_data["gender"]),
            ("TARIKH LAHIR / DATE OF BIRTH:", ic_data["birth_date"]),
            ("AGAMA / RELIGION:", ic_data["religion"]),
            ("WARGANEGARA / NATIONALITY:", ic_data["nationality"]),
            ("TARIKH DIKELUARKAN / DATE OF ISSUE:", ic_data["issue_date"])
        ]
        
        for i, (label, value) in enumerate(fields):
            y_pos = field_y + i * line_height
            draw.text((field_x, y_pos), label, font=font_small, fill='#333333')
            draw.text((field_x, y_pos + 18), value, font=font_regular, fill='#000000')
        
        # Address (multi-line)
        address_y = field_y + len(fields) * line_height + 10
        draw.text((field_x, address_y), "ALAMAT / ADDRESS:", font=font_small, fill='#333333')
        
        # Split address into lines
        address_lines = ic_data["address"].split(", ")
        for i, line in enumerate(address_lines):
            draw.text((field_x, address_y + 18 + i * 22), line, font=font_regular, fill='#000000')
        
        # Add security features simulation
        self._add_security_features(img, draw, width, height)
        
        # Apply realistic distortions
        img = self._apply_realistic_distortions(img)
        
        # Save image
        img.save(output_path, 'JPEG', quality=85)
        return output_path
    
    def _add_security_features(self, img: Image.Image, draw: ImageDraw.Draw, 
                             width: int, height: int):
        """Add simulated security features"""
        # Hologram simulation (semi-transparent overlay)
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Add some geometric patterns
        for i in range(0, width, 50):
            for j in range(0, height, 50):
                overlay_draw.ellipse([i, j, i+20, j+20], fill=(100, 150, 255, 30))
        
        # Merge overlay
        img.paste(Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB'))
        
        # Add fine lines
        for i in range(0, width, 10):
            draw.line([(i, 0), (i, height)], fill='#F0F8FF', width=1)
    
    def _apply_realistic_distortions(self, img: Image.Image) -> Image.Image:
        """Apply realistic camera capture distortions"""
        # Random rotation (-5 to 5 degrees)
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, expand=True, fillcolor='white')
        
        # Random brightness/contrast
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))
        
        # Slight blur
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Add noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 10, (img.height, img.width, 3))
            img_array = np.array(img) + noise
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        return img
    
    def generate_dataset(self, num_samples: int, output_dir: str, 
                        include_annotations: bool = True) -> List[Dict]:
        """Generate a complete synthetic dataset"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        if include_annotations:
            os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
        
        dataset = []
        
        print(f"Generating {num_samples} synthetic Malaysian IC samples...")
        
        for i in range(num_samples):
            # Generate IC data
            ic_data = self.generate_ic_data()
            
            # Create unique filename
            filename = f"synthetic_ic_{i+1:05d}_{uuid.uuid4().hex[:8]}"
            image_path = os.path.join(output_dir, "images", f"{filename}.jpg")
            
            # Create IC image
            self.create_ic_image(ic_data, image_path)
            
            # Create annotation
            annotation = {
                "filename": f"{filename}.jpg",
                "image_path": image_path,
                "fields": ic_data,
                "generated_at": datetime.now().isoformat()
            }
            
            if include_annotations:
                annotation_path = os.path.join(output_dir, "annotations", f"{filename}.json")
                with open(annotation_path, 'w', encoding='utf-8') as f:
                    json.dump(annotation, f, ensure_ascii=False, indent=2)
            
            dataset.append(annotation)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples...")
        
        # Save dataset summary
        summary = {
            "total_samples": num_samples,
            "generated_at": datetime.now().isoformat(),
            "output_directory": output_dir,
            "samples": dataset
        }
        
        summary_path = os.path.join(output_dir, "dataset_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Successfully generated {num_samples} synthetic Malaysian IC samples!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Dataset summary: {summary_path}")
        
        return dataset

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Malaysian IC dataset")
    parser.add_argument("--num_samples", type=int, default=5000, 
                       help="Number of synthetic samples to generate")
    parser.add_argument("--output_dir", type=str, default="synthetic_malaysian_ic_dataset",
                       help="Output directory for generated dataset")
    parser.add_argument("--no_annotations", action="store_true",
                       help="Skip generating annotation files")
    
    args = parser.parse_args()
    
    generator = MalaysianICGenerator()
    dataset = generator.generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        include_annotations=not args.no_annotations
    )
    
    print(f"\n🎉 Dataset generation complete!")
    print(f"Generated {len(dataset)} synthetic Malaysian IC samples")

if __name__ == "__main__":
    main()