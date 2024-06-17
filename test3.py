import cv2
import pytesseract
from pytesseract import Output

# Configuring Tesseract path (if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def detect_and_draw_text(image_path):
   # Đọc ảnh
    img = cv2.imread(image_path)
    
    # Chuyển đổi sang ảnh grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Làm nổi bật vùng chữ
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Làm mịn ảnh để loại bỏ nhiễu
    blurred = cv2.medianBlur(thresh, 5)
    
    # Nhận diện vùng chứa văn bản
    d = pytesseract.image_to_data(blurred, output_type=pytesseract.Output.DICT, config='--psm 6')
    
    # Lưu lại vị trí của các dòng văn bản
    text_regions = []
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # Chỉnh ngưỡng độ tin cậy ở đây
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            text_regions.append((x, y, w, h))
    
    # Vẽ hộp giới hạn cho từng dòng văn bản
    for (x, y, w, h) in text_regions:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Hiển thị ảnh kết quả
    cv2.imshow('Detected Text Regions', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Đường dẫn đến ảnh cần nhận diện
image_path = r'uploads\f3dfeae1-96d2-4c00-bc70-1f267127414a.png'

# Gọi hàm để nhận diện và vẽ hộp giới hạn vùng chứa văn bản
detect_and_draw_text(image_path)