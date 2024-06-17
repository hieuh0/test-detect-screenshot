import cv2
import numpy as np
import pytesseract
import os
from flask import Flask, request, jsonify
import uuid

# Đặt đường dẫn đến Tesseract executable (thay đổi dựa trên hệ thống của bạn)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# Thư mục để lưu trữ ảnh đã được nhận dạng văn bản
SAVE_DIR = 'detected_text_images'

# Dictionary để lưu trạng thái công việc và kết quả
job_queue = {}

# Hàm lưu ảnh cropped vào thư mục
def save_cropped_image(job_id, img, x, y, w, h):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    cropped_img = img[y:y + h, x:x + w]
    filename = f'{job_id}.png'
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, cropped_img)
    print(f'Đã lưu vùng chứa văn bản vào {filepath}')

# Hàm xử lý ảnh và nhận diện vùng chứa văn bản
def detect_and_extract_text(file_name, target_text):
    try:
        # Read the input image
        img = cv2.imread(file_name)

        # Create a grayscale version of the image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, img_thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

        # Dilate the image to enhance text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_dilated = cv2.dilate(img_thresh, kernel, iterations=9)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Initialize variables to store text coordinates
        target_coords = []

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
                # Append coordinates to the list
                target_coords.append((x, y))

                # Draw rectangle around contour on original image
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

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

        return True, target_coords

    except Exception as e:
        print(f'Lỗi xử lý ảnh: {str(e)}')
        return False, []

# Route để upload ảnh
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Không có phần tử file nào'})

        file = request.files['image']
        filename = f'{uuid.uuid4()}.png'
        filepath = os.path.join('uploads', filename)

        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        file.save(filepath)
        print(f'Đã lưu ảnh tải lên vào {filepath}')

        # Tạo job_id duy nhất
        job_id = str(uuid.uuid4())

        # Đoạn văn bản đầu vào cần tìm
        target_text = 'https://en.wikipedia'

        # Gọi hàm detect_and_extract_text để xử lý ảnh và trả về kết quả
        success, target_coords = detect_and_extract_text(filepath, target_text)

        if success:
            job_queue[job_id] = {'status': 'completed', 'result': target_coords}
            return jsonify({'job_id': job_id})
        else:
            job_queue[job_id] = {'status': 'failed', 'error': 'Lỗi xử lý ảnh'}
            return jsonify({'job_id': job_id, 'error': 'Lỗi xử lý ảnh'})

    except Exception as e:
        print(f'Lỗi upload ảnh: {str(e)}')
        return jsonify({'error': str(e)})

# Route để lấy kết quả
@app.route('/get-result/<job_id>', methods=['GET'])
def get_result(job_id):
    try:
        # Trả về thông báo nếu job_id không tồn tại
        if job_id not in job_queue:
            return jsonify({'error': 'ID công việc không hợp lệ'})

        # Trả về trạng thái công việc và kết quả nếu job_id tồn tại
        job_info = job_queue[job_id]

        if job_info['status'] == 'completed':
            return jsonify({'status': 'completed', 'result': job_info['result']})
        elif job_info['status'] == 'failed':
            return jsonify({'status': 'failed', 'error': job_info['error']})
        else:
            return jsonify({'status': 'processing'})

    except Exception as e:
        print(f'Lỗi lấy kết quả: {str(e)}')
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
