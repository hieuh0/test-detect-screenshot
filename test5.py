import cv2
import os
import pytesseract
image = cv2.imread('ac1.png')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

custom_config = r'--oem 3 --psm 6'
d = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)

search_text = 'wikipedia'

n_boxes = len(d['text'])
for i in range(n_boxes):
    if search_text in d['text'][i]:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        print(f'Found text "Why Did My Organic Traffic Drop? Traffic Drop" at location x: {x}, y: {y}')
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Hiển thị ảnh với khung chữ nhật bao quanh văn bản tìm được
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()