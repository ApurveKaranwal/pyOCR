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
from bs4 import BeautifulSoup  # <-- pip install beautifulsoup4

# ---------------- CONFIG ----------------
# NOTE: Update these paths to match your Tesseract and Poppler installations
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


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
    """Enhance image for OCR accuracy: CLAHE, Denoising, and Adaptive Thresholding."""
    if img is None:
        raise ValueError("Image input to preprocess_image is None.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # 2. Improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Adaptive Threshold
    processed = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

    # 4. Morphological closing
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
blood_tests = [
    "hemoglobin", "rbc", "wbc", "platelet", "neutrophil", "lymphocyte",
    "monocyte", "eosinophil", "basophil", "mcv", "mch", "mchc", "rdw",
    "hematocrit", "esr", "bilirubin", "creatinine", "urea", "cholesterol",
    "triglyceride", "hdl", "ldl", "vldl", "glucose", "sugar", "calcium",
    "sodium", "potassium", "chloride", "albumin", "protein", "uric acid",
    "sgot", "sgpt", "alt", "ast", "alkaline phosphatase", "bun", "phosphorus"
]


def extract_blood_data_from_hocr(hocr_data):
    """Parses HOCR data to find blood test names and their nearest numeric values."""
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

        for test in blood_tests:
            if test in text:
                best_match = None
                best_x_distance = float('inf')

                for j in range(i + 1, min(i + 20, len(word_list))):
                    nxt = word_list[j]
                    nx1, ny1, nx2, ny2 = nxt['bbox']
                    if re.match(r'^[\d\.,]+$', nxt['text'].replace(',', '')):
                        if abs((y1 + y2) / 2 - (ny1 + ny2) / 2) < 30:
                            x_distance = nx1 - x2
                            if 0 < x_distance < best_x_distance:
                                best_match = nxt
                                best_x_distance = x_distance

                if best_match:
                    value = best_match['text'].replace(',', '')
                    unit = ""

                    next_idx = word_list.index(best_match) + 1
                    if next_idx < len(word_list):
                        unit_candidate = word_list[next_idx]['text']
                        if (len(unit_candidate) > 0 and
                                not re.match(r'^[\d\.,]+$', unit_candidate) and
                                len(unit_candidate) < 10):
                            unit = unit_candidate

                    results[test.title()] = f"{value} {unit}".strip()
                break
    return results


# ---------------- MAIN PROCESS ----------------
def process_file(path):
    all_hocr = ""

    if path.lower().endswith(".pdf"):
        print("[INFO] Converting PDF pages to images (DPI 500) ...")
        pages = pdf_to_images(path)

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
    output_path = os.path.join(output_dir, f"{base_name}_hocr_output_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Saved results to: {output_path}")
    return output_path


# ---------------- CLI ENTRY ----------------
def main():
    parser = argparse.ArgumentParser(description="Advanced Blood Report OCR to JSON (HOCR-based Accuracy)")
    parser.add_argument("input", help="Path to input PDF or image file")
    args = parser.parse_args()

    try:
        print("--- HOCR-Based OCR Pipeline Starting ---")
        result = process_file(args.input)
        print("\n✅ Final Extracted Data (JSON):\n")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        save_to_json(result, args.input)

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Please ensure the input file exists and Tesseract/Poppler paths are correct.")
    except pytesseract.TesseractNotFoundError:
        print(f"\n[ERROR] Tesseract not found at: {TESSERACT_PATH}")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    main()
