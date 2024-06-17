import cv2
import pytesseract
import numpy as np

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read the image
image = cv2.imread('detected_text_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Apply dilation to fill gaps
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(thresh, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize an empty list to store found text locations
found_locations = []

# Loop through each contour
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)

    # Filter contours based on aspect ratio, area, or dimensions as needed
    if aspect_ratio > 0.5 and w > 10 and h > 10:
        cropped = gray[y:y + h, x:x + w]  # Crop the grayscale image

        # Use Tesseract OCR to extract text
        custom_config = '--oem 3 --psm 6'  # Use PSM 6 for a single block of text
        text = pytesseract.image_to_string(cropped, config=custom_config)

        print(f'Detected text: {text}')

        # Example condition to check if the desired text is found
        if 'Why Did My Organic Traffic Drop? Traffic Drop'.lower() in text.lower():
            found_locations.append({'x': x, 'y': y, 'width': w, 'height': h})
            print(f'Found text "Why Did My Organic Traffic Drop? Traffic Drop" at location x: {x}, y: {y}')

# Print all found locations
print('All found locations:', found_locations)
