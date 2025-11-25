from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

INVOICE_PROMPT = """You are an expert in invoice classification. I will provide you with a block of text that contains the content of an invoice. Your task is:

1. Read the entire invoice text carefully.
2. Analyze the text to determine the type of invoice based on keywords, products, services, or the selling entity.
3. Return only a single line of text: the name of the invoice type.
4. Classify invoices into categories such as:
   - Electricity Invoice
   - Water Invoice
   - Food & Beverage Invoice
   - Consumer Goods Invoice
   - Service Invoice
   - Other Invoice
5. Do not explain your reasoning or add extra text.

Here is the invoice text to classify:
"""
# INVOICE_PROMPT = """Classify the following invoice text into only one category. Return only the category name on a single line. Do not include any explanation or extra text.

# Categories:
# - Electricity Invoice
# - Water Invoice
# - Food & Beverage Invoice
# - Consumer Goods Invoice
# - Service Invoice
# - Other Invoice

# Invoice Text:
# """
def classify_invoice(invoice_text: str) -> str:
    full_prompt = INVOICE_PROMPT + invoice_text

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=full_prompt,
        config=genai.types.GenerateContentConfig(temperature=0.1)
    )

    return response.text.strip()
