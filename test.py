import os
import io
from flask import Flask, request, jsonify
from google import genai
import easyocr
import cv2
import numpy as np
from dotenv import load_dotenv
import json
import re

# Load .env (chứa GOOGLE_API_KEY)
load_dotenv()

# Khởi tạo Gemini Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# EasyOCR Reader
reader = easyocr.Reader(['en', 'vi'])

# Prompt mới để phân loại + nhận tổng tiền
INVOICE_PROMPT = """
You are an expert in invoice classification and invoice amount extraction.
I will provide you with a block of text that contains the content of an invoice.

Your tasks:
1. Read the invoice text carefully.
2. Determine the invoice type.
3. Extract the total amount of the invoice.
4. Return ONLY a valid JSON object. Do NOT use backticks. Do NOT wrap in markdown.

Output format (STRICT):
{
  "invoice_type": "...",
  "total_amount": "..."
}

Classification types (choose one):
- "Electricity Invoice"
- "Water Invoice"
- "Food & Beverage Invoice"
- "Consumer Goods Invoice"
- "Service Invoice"
- "Other Invoice"

Amount rules:
- Prefer the highest monetary value.
- Keep all digits (do NOT shorten or remove zeros).
- Do NOT add currency units.
- Preserve commas/dots if present.

Important:
- Output MUST be raw JSON only.
- Do NOT include any explanation.
- Do NOT include code blocks.
- Do NOT include ```json.
- Do NOT include extra text.

Here is the invoice text to analyze:
"""


# Hàm OCR 1 ảnh
def run_ocr_from_file(file_bytes: bytes) -> str:
    image_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    results = reader.readtext(img)
    text_list = [text for (_, text, _) in results]
    return "\n".join(text_list)

# Hàm xử lý nhiều ảnh
# def run_ocr_multiple(files) -> str:
#     all_text = []
#     for f in files:
#         bytes_data = f.read()
#         all_text.append(run_ocr_from_file(bytes_data))
#     return "\n".join(all_text)

def run_ocr_multiple(files) -> str:
    all_text = []
    for idx, f in enumerate(files):
        bytes_data = f.read()
        text = run_ocr_from_file(bytes_data)
        # Thêm ký tự ngăn cách riêng biệt giữa các file
        all_text.append(f"\n---FILE {idx+1}---\n{text}\n")
    return "\n".join(all_text)

# Hàm gọi Gemini để phân loại + lấy total
def analyze_invoice(full_text: str):
    full_prompt = INVOICE_PROMPT + full_text

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=full_prompt,
        config=genai.types.GenerateContentConfig(temperature=0.1)
    )

    return response.text.strip()

# Flask App
app = Flask(__name__)


# tất cả chữ + tổng + phân loại 
# @app.route("/classify_invoice", methods=["POST"])
# def classify_invoice_api():
#     if "file" not in request.files:
#         return jsonify({"error": "Missing 'file' in request"}), 400

#     files = request.files.getlist("file")

#     try:
#         ocr_text = run_ocr_multiple(files)
#         ai_output = analyze_invoice(ocr_text)

#         # Remove ```json, ``` and other markdown garbage
#         cleaned = ai_output.replace("```json", "").replace("```", "").strip()

#         # Extract JSON block if Gemini added text around it
#         json_match = re.search(r"\{[\s\S]*\}", cleaned)
#         if not json_match:
#             return jsonify({"error": "No valid JSON found from AI", "raw_output": ai_output}), 500

#         cleaned_json = json_match.group(0)

#         # Convert to Python dict
#         result_json = json.loads(cleaned_json)

#         return jsonify({
#             "ocr_text": ocr_text,
#             "ai_output": result_json
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# chỉ tổng + phân loại
@app.route("/classify_invoice", methods=["POST"])
def classify_invoice_api():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in request"}), 400

    files = request.files.getlist("file")

    try:
        # OCR nhiều ảnh
        ocr_text = run_ocr_multiple(files)

        # Gọi gemini
        ai_output = analyze_invoice(ocr_text)

        # Làm sạch backticks
        cleaned = ai_output.replace("```json", "").replace("```", "").strip()

        # Trích JSON
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return jsonify({"error": "No valid JSON found", "raw_output": ai_output}), 500

        json_data = json.loads(match.group(0))
        
        # Chỉ trả về đúng JSON kết quả
        return jsonify(json_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
#


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")