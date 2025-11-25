import os
import io
from flask import Flask, request, jsonify
from google import genai
import easyocr
import cv2
import numpy as np
from dotenv import load_dotenv

# Load .env (chứa GOOGLE_API_KEY)
load_dotenv()

# Khởi tạo Gemini Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# EasyOCR Reader (chỉ khởi tạo 1 lần)
reader = easyocr.Reader(['en', 'vi'])  # thêm 'vi' nếu cần

# Prompt phân loại hóa đơn
INVOICE_PROMPT = """You are an expert in invoice classification. I will provide you with a block of text that contains the content of an invoice. Your task is to:
1. Read the entire invoice text carefully.
2. Analyze the text to determine the type of invoice based on keywords, products, services, or the selling entity.
3. Return only one line of text: the name of the invoice type.
4. Classify invoices into common categories such as:
   - "Electricity Invoice"
   - "Water Invoice"
   - "Food & Beverage Invoice"
   - "Consumer Goods Invoice"
   - "Service Invoice"
   - "Other Invoice"
5. Do not explain your reasoning or add extra text.
Here is the invoice text to classify:
"""

# Hàm OCR từ file upload
def run_ocr_from_file(file_bytes: bytes) -> str:
    image_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    results = reader.readtext(img)
    all_text = [text for (_, text, _) in results]

    return "\n".join(all_text)

# Hàm phân loại với Gemini
def classify_invoice(invoice_text: str) -> str:
    full_prompt = INVOICE_PROMPT + invoice_text
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=full_prompt,
        config=genai.types.GenerateContentConfig(temperature=0.1)
    )
    return response.text.strip()

# Flask app
app = Flask(__name__)

@app.route("/classify_invoice", methods=["POST"])
def classify_invoice_api():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in request"}), 400

    file = request.files["file"]
    try:
        file_bytes = file.read()
        ocr_text = run_ocr_from_file(file_bytes)
        invoice_type = classify_invoice(ocr_text)
        return jsonify({
            "ocr_text": ocr_text,
            "invoice_type": invoice_type
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Chạy Flask với host=0.0.0.0 để ngrok có thể truy cập
    app.run(port=5000, host="0.0.0.0")
