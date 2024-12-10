# app/Controller/routes.py
import os
from flask import Blueprint, render_template, flash, redirect, request, url_for, send_from_directory, current_app
from config import Config
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

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
    if not filename:
        flash('No image uploaded.', 'danger')
        return redirect(url_for('routes.index'))
    
    file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    # Load the image
    image = Image.open(file_path)
    image = image.convert("RGB")  # Ensure the image is in RGB format
    image = image.resize((256, 256))  # Resize to match the input shape of the model
    
    # Convert image to numpy array and normalize to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Load the TensorFlow Lite model
    model_path = os.path.join(Config.ROOT_PATH, 'MachineLearning', 'potato_model_4_11_21.tflite')
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor and invoke the interpreter
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    # Get the output from the model
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Debugging: Inspect model output
    print("Model output:", output_data)
    
    # Process the output to make it usable for display
    output_img = np.reshape(output_data, (256, 256))  # Assuming the model output is a 2D image
    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min()) * 255.0  # Normalize to [0, 255]
    output_img = output_img.astype(np.uint8)  # Convert to 8-bit integers
    
    # Optionally, apply a threshold to highlight certain features
    _, result_img = cv2.threshold(output_img, 50, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed
    
    # Convert the result back to an image for display
    result_pil = Image.fromarray(result_img)

    # Ensure the image is in a format compatible with JPEG
    if result_pil.mode != "RGB":
        result_pil = result_pil.convert("RGB")  # Convert to RGB mode

    # Save the result image
    result_filename = f"result_{filename}"
    result_path = os.path.join(Config.RESULT_FOLDER, result_filename)
    result_pil.save(result_path, format="JPEG")
    
    # Return results to the template
    return render_template('index.html', title="Tuber Ruler", result_filename=result_filename)

@bp_routes.route('/history', methods=['GET'])
def history():
    return render_template('history.html', title="History")

@bp_routes.route('/settings', methods=['GET'])
def settings():
    return render_template('settings.html', title="Settings")