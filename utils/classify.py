from google import genai
import os
from dotenv import load_dotenv
from fetch_categories import classification_text

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

INVOICE_PROMPT = """You are an expert in invoice classification. I will provide you with a block of text that contains the content of an invoice. Your task is:
 
1. Read the entire invoice text carefully.
2. Analyze the text to determine the type of invoice based on keywords, products, services, or the selling entity.
3. Return only a single line of text: the name of the invoice type.
4. Classification types (choose ONLY ONE from the list below):
{classification_text}

5. Do not explain your reasoning or add extra text.
 
Here is the invoice text to classify:
"""
def classify_invoice(invoice_text: str) -> str:
    # Thêm .format() để thay thế {classification_text}
    formatted_prompt = INVOICE_PROMPT.format(classification_text=classification_text)
    full_prompt = formatted_prompt + invoice_text
  
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=full_prompt,
        config=genai.types.GenerateContentConfig(temperature=0.1)
    )

    return response.text.strip()