# Description: This file contains the code for the server that will be used to upload the audio files to the server.
from flask import Flask, request, jsonify
from flask_cors import CORS
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

    # Clean up after processing the file (remove the file)
    os.remove(file_path)
    if file_path.endswith('.mp3') and os.path.exists(file_path.replace('.mp3', '.wav')):
        os.remove(file_path.replace('.mp3', '.wav'))

# Run the app
if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True)