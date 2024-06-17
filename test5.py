import cv2
import os
import pytesseract

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_and_extract_text(file_name, target_text):
    # Read the input image
    img = cv2.imread(file_name)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to create a binary image
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Dilate the image to enhance text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_dilated = cv2.dilate(img_thresh, kernel, iterations=1)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store text coordinates
    target_coords = []

    # Process each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter contours based on size (adjust as needed)
        if w < 10 or h < 10:
            continue
        
        # Crop the region of interest (ROI) from the original image
        roi = img[y:y+h, x:x+w]

        # Use Tesseract OCR to extract text from the ROI
        custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode (OEM) 3 for default, Page Segmentation Mode (PSM) 6 for a single block of text
        text = pytesseract.image_to_string(roi, config=custom_config)
        
        # Check if target text is found
        if target_text.lower() in text.lower():
            # Append coordinates to the list
            target_coords.append((x, y))

            # Draw rectangle around contour on original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow('Detected Text', img)
    cv2.waitKey(0)

    # Save the image with bounding boxes
    output_dir = 'output'  # Specify the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, os.path.basename(file_name))
    cv2.imwrite(output_file, img)
    print(f'Saved image with bounding boxes to: {output_file}')

    # Print target coordinates
    for i, coords in enumerate(target_coords):
        print(f'Target Text {i+1} Coordinates: {coords}')

# Example usage:
file_name = r'ac1.png'
target_text = 'Human Traffic'
detect_and_extract_text(file_name, target_text)
