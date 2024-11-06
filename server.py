# Description: This file contains the code for the server that will be used to upload the audio files to the server.
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import librosa  # Import librosa for BPM detection
from pydub import AudioSegment
import numpy as np


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

    try:
        # Check if file format needs conversion
        if file_path.endswith(".mp3"):
            # Convert .mp3 to .wav using pydub
            audio = AudioSegment.from_mp3(file_path)
            file_path_wav = file_path.replace(".mp3", ".wav")
            audio.export(file_path_wav, format="wav")
            y, sr = librosa.load(file_path_wav)
            os.remove(file_path_wav)  # Clean up the .wav file after loading
        else:
            y, sr = librosa.load(file_path)

        # Log to confirm file loading
        print("File loaded successfully. Starting BPM detection.")

       # Use librosa's beat_track function to estimate tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=90, tightness=100)
        
       # Ensure tempo is a single scalar by taking the first element if it's an array
        bpm = round(float(tempo[0]), 2) if isinstance(tempo, np.ndarray) else round(tempo, 2)

       # Filter out unreasonable values
        if not (40 <= bpm <= 200):
            bpm = "BPM detection failed or is unreliable"

    except Exception as e:
        # Log the specific error details for debugging
        print("Error analyzing file:", str(e))
        os.remove(file_path)
        return jsonify({'error': 'Error analyzing file', 'details': str(e)}), 500
    
    os.remove(file_path)
    return jsonify({'message': 'File uploaded successfully', 'bpm': bpm}), 200

# Run the app
if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True, host="0.0.0.0", port=5000)