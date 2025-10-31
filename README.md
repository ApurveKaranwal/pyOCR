# 🧠 pyOCR

A Python-based Optical Character Recognition (OCR) tool that extracts text from **PDFs** and **images** using **Tesseract OCR**, **OpenCV**, and **Poppler**.  
It intelligently processes files, cleans extracted data, and outputs structured text in **JSON** format.

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
---

## 🚀 Features
- 🖼️ Convert **images** and **PDFs** to readable text  
- ⚙️ Uses **OpenCV** for image preprocessing (noise reduction, thresholding, etc.)  
- 🔍 Extracts text using **Tesseract OCR**  
- 📄 Converts multi-page PDFs with **pdf2image**  
- 🧹 Cleans and structures extracted text using **regex**  
- 💾 Outputs text and metadata in **JSON** format  

---

## 🧩 Tech Stack
- **Python 3.12**
- **OpenCV**
- **Pytesseract**
- **pdf2image**
- **BeautifulSoup**
- **PIL (Pillow)**
- **Poppler**
- **re** and **json** for text processing

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/pyOCR.git
cd pyOCR
```
### 2. Install Dependencies
```bash
pip install opencv-python pytesseract pdf2image pillow
```

### 3. Install Tesseract and Poppler

### Windows
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler: https://github.com/oschwartz10612/poppler-windows?tab=readme-ov-file

### Linux
```bash
sudo apt install tesseract-ocr poppler-utils
```

### 4. Run the Script
```bash
python pyOCR.py "blood_report.pdf"
```
---

## Output Example
```json
{
    "Hemoglobin": "3 window",
    "Rbc": "4.19 million/emm",
    "Hematocrit": "43.3 %",
    "Mcv": "90.3 fl",
    "Mch": "33.4 g/dl",
    "Rdw": "13.60 %",
    "Wbc": "1 10570",
    "Neutrophil": "13 %",
    "Lymphocyte": "19 %",
    "Eosinophil": "02 %",
    "Monocyte": "06 %",
    "Platelet": "190000 /cmm",
    "Esr": "7 mm/1hr",
    "Cholesterol": "3",
    "Triglyceride": "168.0 mg/dl",
    "Ldl": "33.60 mg/dl",
    "Hdl": "1.7 up",
    "Sugar": "141.0 mg/dl",
    "Glucose": "1 c",
    "Albumin": "4.20 g/dl",
    "Creatinine": "0.83 mg/dl",
    "Protein": "7.00 g/dl",
    "Bilirubin": "0.20 mg/dl",
    "Urea": "8.41 mg/dl",
    "Uric Acid": "4.90 mg/dl",
    "Calcium": "25",
    "Sgpt": "48.0 u/l",
    "Sgot": "27.0 u/l",
    "Sodium": "143.00 mmol/l",
    "Potassium": "4.90 mmol/l",
    "Chloride": "105.0 mmol/l"
}
```

---

## 🧠 How It Works
- Converts PDF pages to images (if input is a PDF).
- Applies OpenCV preprocessing for better text clarity.
- Runs Tesseract OCR to extract readable text.
- Cleans and structures the text using regular expressions.
- Exports the final data to a JSON file.

---

## 🤝 Contributing
Pull requests are welcome!
If you’d like to improve the tool or add features, feel free to fork the repo and submit a PR.

---

## ⭐ If you like this project, consider giving it a star on GitHub!

---

## 👤 Author
- Apurve Karanwal
- linkedin: https://www.linkedin.com/in/apurvekaranwal
- x: https://x.com/Apurve_Karanwal

