from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

# Import CORS from flask_cors
app = Flask(__name__)
# Add CORS to the app
CORS(app)
# Set the upload folder
app.config['UPLOAD_FOLDER'] = './uploads'
# Create the upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the upload route for the server
@app.route('/upload', methods=['POST'])
# Define the upload_file function
def upload_file():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    # Get the file from the request
    file = request.files['file']
    # Check if the file is empty or not
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file to the upload folder
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)