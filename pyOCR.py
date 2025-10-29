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

# ---------------- CONFIG ----------------
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(img):
    """Enhance image for OCR accuracy"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast using CLAHE (better for scanned docs)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise and smooth
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Adaptive threshold for better binarization
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 15
    )

    # Remove small noise and join broken text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Invert (black text on white)
    processed = 255 - morph
    return processed


# ---------------- PDF HANDLING ----------------
def pdf_to_images(pdf_path):
    """Convert PDF pages to list of images"""
    return convert_from_path(pdf_path, dpi=400, poppler_path=POPPLER_PATH)


# ---------------- OCR FUNCTION ----------------
def image_to_text(image):
    """Perform OCR with better table/line detection"""
    custom_config = r'--oem 3 --psm 6'  # OEM 3 = Best accuracy, PSM 6 = Assume uniform blocks of text
    text = pytesseract.image_to_string(image, lang='eng', config=custom_config)
    return text


# ---------------- BLOOD TEST EXTRACTION ----------------
blood_tests = [
    "hemoglobin", "rbc", "wbc", "platelet", "neutrophil", "lymphocyte",
    "monocyte", "eosinophil", "basophil", "mcv", "mch", "mchc", "rdw",
    "hematocrit", "esr", "bilirubin", "creatinine", "urea", "cholesterol",
    "triglyceride", "hdl", "ldl", "vldl", "glucose", "sugar", "calcium",
    "sodium", "potassium", "chloride", "albumin", "protein", "uric acid",
    "sgot", "sgpt", "alkaline phosphatase", "bun", "phosphorus"
]


def extract_blood_data(text):
    """Extract only blood test components and values (handles tabular data too)"""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    results = {}

    for line in lines:
        lower_line = line.lower()

        for test in blood_tests:
            if test in lower_line:
                # Extract the test value next to the name, even if in table form
                match = re.search(rf"{test}[^0-9]*([0-9]+\.?[0-9]*)\s*([a-zA-Z%/µ\d]*)", lower_line)
                if not match:
                    # fallback for numeric column structure
                    match = re.search(r"([0-9]+\.?[0-9]*)\s*([a-zA-Z%/µ\d]*)", line)
                if match:
                    value = f"{match.group(1)} {match.group(2)}".strip()
                    results[test.title()] = value
                break

    return results


# ---------------- MAIN PROCESS ----------------
def process_file(path):
    all_text = ""

    if path.lower().endswith(".pdf"):
        pages = pdf_to_images(path)
        for i, page in enumerate(pages, 1):
            print(f"[INFO] Processing page {i}/{len(pages)} ...")
            img = np.array(page)
            processed = preprocess_image(img)
            text = image_to_text(processed)
            all_text += "\n" + text
    else:
        img = cv2.imread(path)
        processed = preprocess_image(img)
        text = image_to_text(processed)
        all_text = text

    blood_data = extract_blood_data(all_text)
    return blood_data


# ---------------- FILE SAVING ----------------
def save_to_json(data, input_path):
    """Save OCR results as JSON next to input file"""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_output_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Saved results to: {output_path}")
    return output_path


# ---------------- CLI ENTRY ----------------
def main():
    parser = argparse.ArgumentParser(description="Advanced Blood Report OCR to JSON (Improved Accuracy)")
    parser.add_argument("input", help="Path to input PDF or image file")
    args = parser.parse_args()

    print("[INFO] Starting enhanced OCR pipeline ...")
    result = process_file(args.input)

    print("\n✅ Final Extracted Data (JSON):\n")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    save_to_json(result, args.input)


if __name__ == "__main__":
    main()
