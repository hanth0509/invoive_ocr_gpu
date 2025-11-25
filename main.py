import warnings
import time
import torch
from utils.ocr_reader import run_ocr
from utils.classify import classify_invoice

# Ẩn các cảnh báo liên quan đến thư viện
# warnings.filterwarnings("ignore")
# if torch.cuda.is_available():
#     torch.backends.cudnn.enabled=True
def main():
    image_path = "images/sample_invoice.jpg"  # đổi theo file của bạn
# 1. Ghi lại thời điểm bắt đầu đo
    start_time = time.time()
    print("🔍 Running OCR...")
    text = run_ocr(image_path)
    print("\n===== OCR TEXT =====")
    print(text)

    print("\n🤖 Classifying invoice type...")
    invoice_type = classify_invoice(text)
    print("\n🎯 Invoice Type:", invoice_type)
    
    # 2. Ghi lại thời điểm kết thúc đo
    end_time = time.time()
    
    # 3. Tính toán và in ra tổng thời gian
    elapsed_time = end_time - start_time
    
    # In thời gian, làm tròn đến 2 chữ số thập phân
    print(f"\n==========================================")
    print(f"⏱️ Total Execution Time (OCR + Classification): {elapsed_time:.2f} seconds")
    print(f"==========================================")
if __name__ == "__main__":
    main()
