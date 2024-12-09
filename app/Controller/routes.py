# app/Controller/routes.py
import os
from flask import Blueprint, render_template, flash, redirect, request, url_for, send_from_directory, current_app
from config import Config
from werkzeug.utils import secure_filename

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

@bp_routes.route('/go', methods=['GET'])
def go():
    return render_template('index.html', title="Tuber Ruler")

@bp_routes.route('/history', methods=['GET'])
def history():
    return render_template('history.html', title="History")

@bp_routes.route('/settings', methods=['GET'])
def settings():
    return render_template('settings.html', title="Settings")