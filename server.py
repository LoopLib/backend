# Description: This file contains the code for the server that will be used to upload the audio files to the server.
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import librosa  # Import librosa for BPM detection

# Import CORS from flask_cors
app = Flask(__name__)
# Add CORS to the app
CORS(app, resources={r"/upload": {"origins": "*"}})  # Allow all origins for /upload
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
    
    # Log the uploaded file name to the server console
    print("Received file:", file.filename)
    
    # Save the file to the upload folder
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Detect BPM using librosa
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_path)
        # Calculate the onset strength using librosa
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Detect the BPM using librosa
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)  # Convert to a single float value if necessary
    except Exception as e:
        os.remove(file_path)
        return jsonify({'error': 'Error analyzing file', 'details': str(e)}), 500

    # Clean up after processing the file (remove the file)
    os.remove(file_path)

    # Send a success response back to the client, including the BPM
    return jsonify({'message': 'File uploaded successfully', 'bpm': bpm}), 200

# Run the app
if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True, host="0.0.0.0", port=5000)