import easyocr
import cv2
from matplotlib import pyplot as plt
# Khởi tạo EasyOCR Reader (chỉ 1 lần)
# reader = easyocr.Reader(['en', 'vi'])
reader = easyocr.Reader(['en'])

def run_ocr(image_path: str) -> str:
    """Trích xuất text từ ảnh và vẽ bounding box (chỉ để xem trực quan)."""
    results = reader.readtext(image_path)
    all_text = []

    img = cv2.imread(image_path)
    for (bbox, text, confidence) in results:
        all_text.append(text)
        # Vẽ bounding box
        # top_left = tuple(map(int, bbox[0]))
        # bottom_right = tuple(map(int, bbox[2]))
        # cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        # cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Hiển thị ảnh
    # plt.figure(figsize=(12, 8))
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    return '\n'.join(all_text)
