import cv2
import pytesseract

# Load the image from file
image = cv2.imread('hieu_test.png')

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary inverse thresholding to the grayscale image
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Define a kernel for dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

# Apply dilation to merge text regions
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Find contours in the dilated image
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Text to search for
search_text = 'Human Traffic'

# List to store the coordinates of matched text regions
matched_coords = []
# Iterate over each contour and draw a bounding box
for contour in contours:
    # x, y, w, h = cv2.boundingRect(contour)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(contour)
    # Extract the region of interest (ROI) from the grayscale image
    roi = gray[y:y + h, x:x + w]
    
    # Use Tesseract to extract text from the ROI
    extracted_text = pytesseract.image_to_string(roi, config='--oem 3 --psm 6')
    
    # Check if the search text is in the extracted text
    if search_text.lower() in extracted_text.lower():
        matched_coords.append((x, y, w, h))
        # Draw a rectangle around the matched text region
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f'Found text "{search_text}" at location x: {x}, y: {y}')

# Save the image with rectangles
cv2.imwrite('annotated_image.png', image)

# Display the image with rectangles
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
