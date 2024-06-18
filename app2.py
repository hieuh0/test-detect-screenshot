import cv2
import numpy as np
import pytesseract
import os
from flask import Flask, request, jsonify
import uuid
import time
from concurrent.futures import ThreadPoolExecutor

#Set path to Tesseract executable (varies based on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# Thư mục để lưu trữ ảnh đã được nhận dạng văn bản
SAVE_DIR = 'detected_text_images'

#Dictionary to save job status and results
job_queue = {}

#ThreadPoolExecutor for multithreading
executor = ThreadPoolExecutor()

# Hàm lưu ảnh cropped vào thư mục
def save_cropped_image(job_id, img, x, y, w, h):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    cropped_img = img[y:y + h, x:x + w]
    filename = f'{job_id}.png'
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, cropped_img)
    print(f'Text container saved to {filepath}')

# Hàm xử lý ảnh và nhận diện vùng chứa văn bản
def detect_and_extract_text_async(job_id, file_name, target_text):
    try:
        # Read the input image
        img = cv2.imread(file_name)

        # Create a grayscale version of the image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

        # Define a kernel for dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        # Apply dilation to merge text regions
        img_dilated = cv2.dilate(img_thresh, kernel, iterations=1)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to store text coordinates
        target_coords = None

        # Process each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 35 and h < 35:
                continue
            
            # Crop the region of interest (ROI) from the original image
            roi = img[y:y+h, x:x+w]

            # Use Tesseract OCR to extract text from the ROI
            custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode (OEM) 3 for default, Page Segmentation Mode (PSM) 6 for a single block of text
            text = pytesseract.image_to_string(roi, config=custom_config)
            
            # Check if target text is found
            if target_text.lower() in text.lower():
                # Save the cropped image
                save_cropped_image(job_id, img, x, y, w, h)

                # Store the coordinates
                target_coords = {'x': x, 'y': y}
                break  # Stop processing after first match

        if target_coords:
            job_queue[job_id] = {'status': 'completed', 'result': target_coords}
        else:
            job_queue[job_id] = {'status': 'failed', 'error': 'text to be recognized could not be found'}

    except Exception as e:
        job_queue[job_id] = {'status': 'failed', 'error': f'Image processing error: {str(e)}'}

# Route để upload ảnh
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files and "text_to_find" not in request.form:
            return jsonify({'error': 'invalid param', 'status': 'failed'})

        file = request.files['image']
        filename = f'{uuid.uuid4()}.png'
        filepath = os.path.join('uploads', filename)

        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        file.save(filepath)
        print(f'Uploaded photo saved to {filepath}')

        job_id = str(uuid.uuid4())

        target_text = request.form['text_to_find']

        # Save job information to job_queue
        job_queue[job_id] = {'status': 'processing'}

        # Execute the detect_and_extract_text_async function asynchronously
        executor.submit(detect_and_extract_text_async, job_id, filepath, target_text)

        return jsonify({'job_id': job_id, 'status': 'completed'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'})

# Route to get results
@app.route('/get-result/<job_id>', methods=['GET'])
def get_result(job_id):
    try:
        if job_id not in job_queue:
            return jsonify({'status': 'failed', 'error': 'Invalid job ID'})
        job_info = job_queue[job_id]

        if job_info['status'] == 'completed':
            result = job_info['result']
            return jsonify({'result': result, 'status': 'completed'})
        elif job_info['status'] == 'failed':
            return jsonify({'status': 'failed', 'error': job_info['error']})
        else:
            return jsonify({'status': 'processing'})

    except Exception as e:
        return jsonify({'status': 'failed', 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
