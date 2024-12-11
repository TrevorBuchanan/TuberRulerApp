# app/Controller/routes.py
import os
from flask import Blueprint, render_template, flash, redirect, request, url_for, send_from_directory, current_app
from config import Config
from werkzeug.utils import secure_filename
from MachineLearning.potato_ml import main
import cv2
import numpy as np


bp_routes = Blueprint('routes', __name__, template_folder=Config.TEMPLATE_FOLDER)

@bp_routes.route('/', methods=['GET'])
def index():
    return render_template('index.html', title="Tuber Ruler")

@bp_routes.route('/open-camera', methods=['GET'])
def open_camera():
    return render_template('index.html', title="Tuber Ruler", camera_capture_enabled=True)

# Utility function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp_routes.route('/upload', methods=['GET', 'POST'])
def upload():
    filename = None  # Initialize filename to None in case the file is not uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Sanitizing the filename
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            
            # Save the file
            file.save(file_path)
            
            flash('Upload successful!', 'success')
            return render_template('index.html', title="Tuber Ruler", image_filename=filename)
        else:
            flash('Invalid file type. Only image files are allowed.', 'danger')
            return redirect(request.url)
    flash('Upload unsuccessful!', 'failure')
    return redirect(url_for('routes.upload'))  

@bp_routes.route('/uploaded-file/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename)

@bp_routes.route('/result-file/<filename>', methods=['GET'])
def result_file(filename):
    return send_from_directory(Config.RESULT_FOLDER, filename)

@bp_routes.route('/go/<filename>', methods=['GET'])
def go(filename):
    # Define paths for input and output
    input_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    output_filename = f"result_{filename}"
    output_path = os.path.join(Config.RESULT_FOLDER, output_filename)
    
    try:
        # Read the image data
        input_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        # Extract results from the function's output (example, may vary based on actual data)
        coin_string = "Quarter"  # Change this to the desired coin type (e.g., "Dime" or "C_2.5")
        potato_arr, num_str, min_str, max_str, avg_str, buff_img = main(input_img, coin_string)
        
        potato_list = potato_arr.tolist()
        # print(f"Potatoes: {potato_arr}")
        # print(f"Num str: ({num_str})")
        # print(f"Min str: {min_str}")
        # print(f"Max str: {max_str}")
        # print(f"Avg str: {avg_str}")
        
        # Convert the byte buffer to an image
        image_array = np.frombuffer(buff_img, dtype=np.uint8)
        result_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Save the result image to a file
        cv2.imwrite(output_path, result_img)
        return render_template('index.html', 
                               title="Tuber Ruler", 
                               result_filename=output_filename,
                               num_potatoes=num_str,
                               min_len_wid=min_str,
                               max_len_wid=max_str,
                               average_len_wid=avg_str,
                               potato_list=potato_list)
    except Exception as e:
        flash(f"Error processing image: {e}", 'danger')
        print(f"Error processing image: {e}")
        return redirect(url_for('routes.index'))

@bp_routes.route('/history', methods=['GET'])
def history():
    return render_template('history.html', title="History")

@bp_routes.route('/settings', methods=['GET'])
def settings():
    return render_template('settings.html', title="Settings")