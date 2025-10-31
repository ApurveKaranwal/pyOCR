import os
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import re
import json
import argparse
from datetime import datetime
from bs4 import BeautifulSoup
import sys 

# ---------------- CONFIG ----------------
# NOTE: Update these paths to match your Tesseract and Poppler installations
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"

try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except pytesseract.TesseractNotFoundError:
    print(f"\n[FATAL ERROR] Tesseract not found at: {TESSERACT_PATH}")
    print("Please update TESSERACT_PATH in the script.")
    sys.exit(1)


# ---------------- BLOOD TEST MAPPING ----------------
# Comprehensive map with common OCR errors added for better coverage.
blood_tests_map = {
    "Hemoglobin": ["hemoglobin", "hgb", "hb", "hemeglobin"],
    "PCV": ["pcv", "packed cell volume", "hematocrit"],
    "RBC Count": ["rbc", "red blood cell count", "mount"], 
    "MCV": ["mcv", "mev"],
    "MCH": ["mch"],
    "MCHC": ["mchc"],
    "RDW": ["rdw", "red cell distribution width"],
    "TLC": ["tlc", "total leukocyte count", "wbc"],
    "Neutrophils": ["neutrophil", "segmented neutrophils", "neuophile", "neunrophils"],
    "Lymphocytes": ["lymphocyte", "lyphocytes", "lymph"],
    "Monocytes": ["monocyte"],
    "Eosinophils": ["eosinophil", "eosinphils"],
    "Basophils": ["basophil"],
    "Absolute Neutrophil Count": ["absolute", "neuophile"],
    "Absolute Lymphocyte Count": ["absolute", "lyphocytes", "lyphocyres"],
    "Absolute Eosinophil Count": ["absolute", "eosinophil"],
    "Absolute Basophil Count": ["absolute", "basophil"],
    "Platelet Count": ["platelet", "thrombocyte", "platelet count"],
    "MPV": ["mpv", "mean platelet volume"],
    "Bilirubin": ["bilirubin"], "Creatinine": ["creatinine"],
    "Glucose": ["glucose", "sugar"], "Cholesterol": ["cholesterol"],
    "Triglyceride": ["triglyceride"], "Uric Acid": ["uric acid"],
    "SGOT/AST": ["sgot", "ast"], "SGPT/ALT": ["sgpt", "alt"],
    "Albumin": ["albumin"], "Protein": ["protein"],
}

# ---------------- IMAGE PREPROCESSING ----------------
def correct_skew(img):
    """Attempt to correct rotation (skew) using Tesseract's orientation data."""
    try:
        osd = pytesseract.image_to_osd(Image.fromarray(img))
        rotation_match = re.search(r'Rotate: (\d+)', osd)
        if rotation_match:
            angle = int(rotation_match.group(1))
            if angle != 0:
                print(f"[INFO] Deskewing detected angle: {angle} degrees.")
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, -angle, 1.0) 
                img = cv2.warpAffine(
                    img, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
        return img
    except Exception:
        return img


def preprocess_image(img):
    """Enhance image for OCR accuracy: Denoising, CLAHE, and Adaptive Thresholding."""
    if img is None:
        raise ValueError("Image input to preprocess_image is None.")
        
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img 

    gray = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    processed = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)

    return processed


# ---------------- PDF HANDLING ----------------
def pdf_to_images(pdf_path):
    """Convert PDF pages to list of images with high DPI for better detail."""
    return convert_from_path(pdf_path, dpi=500, poppler_path=POPPLER_PATH)


# ---------------- OCR FUNCTION (HOCR) ----------------
def image_to_hocr(image):
    """Perform OCR and return HOCR data for coordinate-based parsing."""
    custom_config = r'--oem 3 --psm 6 -c tessedit_create_hocr=1'
    hocr_data = pytesseract.image_to_pdf_or_hocr(image, extension='hocr', config=custom_config)
    return hocr_data.decode('utf-8')


# ---------------- BLOOD TEST EXTRACTION (HOCR-BASED) ----------------
# Helper function to clean OCR results
def clean_numeric_value(text, test_name=""):
    """
    Strips non-numeric/decimal characters and tries to fix missing decimals
    based on typical report values.
    """
    # Strips common OCR junk and cleans text
    text = text.lower().replace(',', '').replace(' ', '').strip('|>.,»:') 
    
    # Extract all digit/decimal patterns
    match = re.search(r'^[<>]?(\d*\.?\d+)', text)
    if match:
        cleaned_value = match.group(1)
        
        # Aggressive cleaning for common report issues (e.g., 4500 -> 45.00 for percentages)
        if '.' not in cleaned_value and len(cleaned_value) > 3 and 'hgb' not in test_name.lower():
            cleaned_value = cleaned_value[:-2] + '.' + cleaned_value[-2:]
        
        return cleaned_value
    
    # If it's not a number but a dash or other non-numeric result
    if text in ["-", "---", "negative", "trace"]:
        return text
        
    return "" 

def extract_blood_data_from_hocr(hocr_data):
    """Parses HOCR data to find blood test names and their nearest numeric values/units."""
    soup = BeautifulSoup(hocr_data, 'html.parser')
    results = {}

    words = soup.find_all('span', class_='ocrx_word')
    word_list = []
    for w in words:
        title = w.get('title', '')
        bbox = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
        if bbox:
            x1, y1, x2, y2 = map(int, bbox.groups())
            word_list.append({
                'text': w.text.strip().lower(),
                'bbox': (x1, y1, x2, y2)
            })

    for i, wd in enumerate(word_list):
        text = wd['text']
        x1, y1, x2, y2 = wd['bbox']
        test_display_name = None

        # 1. Identify the Test Name
        for display_name, search_terms in blood_tests_map.items():
            if any(term in text for term in search_terms):
                test_display_name = display_name
                break 

        if test_display_name and test_display_name not in results:
            
            best_match = None
            best_x_distance = float('inf') 

            # 2. Search for the Numeric Result (Search 30 words forward)
            for j in range(i + 1, min(i + 30, len(word_list))):
                nxt = word_list[j]
                nx1, ny1, nx2, ny2 = nxt['bbox']
                
                # Check for numerical value 
                is_value = re.match(r'^[<>]?[\d\.,-/\s]+$', nxt['text'].replace(',', '').strip('|>'))
                
                # Loosened vertical alignment tolerance (80 pixels) - FINAL SETTING
                if is_value and abs((y1 + y2) / 2 - (ny1 + ny2) / 2) < 80:
                    x_distance = nx1 - x2 
                    # Prioritize the number with the smallest horizontal gap after the test name
                    if x_distance >= 0 and x_distance < best_x_distance:
                        best_match = nxt
                        best_x_distance = x_distance

            # 3. Process and Clean the Result + Extended Unit Search (Strict on Proximity)
            if best_match:
                value = clean_numeric_value(best_match['text'], test_display_name)
                unit = ""
                
                # Search up to 5 words after the result for the Unit
                for k in range(word_list.index(best_match) + 1, min(word_list.index(best_match) + 6, len(word_list))):
                    unit_candidate = word_list[k]
                    unit_text = unit_candidate['text']
                        
                    # Unit check: Must be on the same vertical line, short, non-numeric, AND horizontally CLOSE
                    if (abs((y1 + y2) / 2 - (unit_candidate['bbox'][1] + unit_candidate['bbox'][3]) / 2) < 80 and 
                        unit_candidate['bbox'][0] - best_match['bbox'][2] < 150 and # Horizontal proximity
                        len(unit_text) > 0 and 
                        len(unit_text) < 10 and 
                        not re.match(r'^[\d\.,]+$', unit_text)):
                            
                            # Final Check: Accept any short text that isn't clearly random junk characters (like 'bb')
                            if not re.search(r'[\!@\#\$\%\^\&\*`~]', unit_text) and unit_text not in ['be', 'bb', 'ki', 'thouknns', 'and']: 
                                unit = unit_text
                                break 

                results[test_display_name] = f"{value} {unit}".strip()
                continue 
                
    return results


# ---------------- MAIN PROCESS ----------------
def process_file(path):
    all_hocr = ""
    if path.lower().endswith(".pdf"):
        print("[INFO] Converting PDF pages to images (DPI 500) ...")
        pages = convert_from_path(path, dpi=500, poppler_path=POPPLER_PATH)
        for i, page in enumerate(pages, 1):
            print(f"[INFO] Processing page {i}/{len(pages)} ...")
            img = np.array(page)
            img = correct_skew(img)
            processed = preprocess_image(img)
            hocr = image_to_hocr(processed)
            all_hocr += hocr
    else:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image file: {path}")
        print("[INFO] Processing image ...")
        img = correct_skew(img)
        processed = preprocess_image(img)
        all_hocr = image_to_hocr(processed)

    data = extract_blood_data_from_hocr(all_hocr)
    return data


# ---------------- FILE SAVING ----------------
def save_to_json(data, input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_extracted_data_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Saved results to: {output_path}")
    return output_path


# ---------------- CLI ENTRY ----------------
def main():
    parser = argparse.ArgumentParser(description="Advanced Blood Report OCR to JSON (HOCR-based Accuracy)")
    parser.add_argument("input", help="Path to input PDF or image file (e.g., main_notred.jpg)")
    args = parser.parse_args()

    try:
        print("--- HOCR-Based OCR Pipeline Starting ---")
        result = process_file(args.input)
        print("\n✅ Final Extracted Data (JSON):\n")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        save_to_json(result, args.input)

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Please ensure the input file exists.")
    except pytesseract.TesseractNotFoundError:
        print(f"\n[ERROR] Tesseract not found at: {TESSERACT_PATH}")
        print("Please verify the TESSERACT_PATH in the script.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()