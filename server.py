# Description: This file contains the code for the server that will be used to upload the audio files to the server.
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import librosa  # Import librosa for BPM detection
from pydub import AudioSegment
import numpy as np
from detection import detect_key  # Import detect_key function from detection.py

# Import CORS from flask_cors
app = Flask(__name__)
# Add CORS to the app
CORS(app, resources={r"/upload": {"origins": "*"}})  # Allow all origins for /upload
# Set the upload folder
app.config['UPLOAD_FOLDER'] = './uploads'
# Create the upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function to safely extract scalar from array or sequence
def to_scalar(value):
    if isinstance(value, np.ndarray):
        return float(value[0]) if value.size > 0 else 0.0
    return float(value)
        
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
        # Convert .mp3 to .wav if necessary
        if file_path.endswith(".mp3"):
            audio = AudioSegment.from_mp3(file_path)
            file_path_wav = file_path.replace(".mp3", ".wav")
            audio.export(file_path_wav, format="wav")
            y, sr = librosa.load(file_path_wav)
            os.remove(file_path_wav)
        else:
            y, sr = librosa.load(file_path)

        print("File loaded successfully. Starting BPM detection.")

        # Step 1: Trim silence from start and end and normalize volume
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Step 2: Take a stable segment (e.g., middle of track) for consistent BPM
        middle_start = len(y) // 4
        middle_end = 3 * len(y) // 4
        y = y[middle_start:middle_end]

        # Step 3: Harmonic-Percussive Separation to isolate percussive elements
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

        # Primary tempo detection using beat_track (standard method)
        tempo_beat, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=onset_env, start_bpm=90, tightness=100)
        tempo_beat = to_scalar(tempo_beat)

        # Secondary tempo detection using librosa.feature.rhythm.tempo
        tempos = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
        tempo_onset = to_scalar(np.median(tempos))

        # Tertiary: Alternative parameters to capture other rhythm patterns
        tempo_beat_alt, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=onset_env, start_bpm=60, tightness=80)
        tempo_beat_alt = to_scalar(tempo_beat_alt)

        # Fine detection using smaller window size for high temporal resolution
        hop_length = 256  # smaller window
        onset_env_fine = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)
        tempos_fine = librosa.feature.rhythm.tempo(onset_envelope=onset_env_fine, sr=sr, hop_length=hop_length)
        tempo_fine = to_scalar(np.median(tempos_fine))

        # Aggregate all detected tempos with weighted voting
        bpm_estimates = [tempo_beat, tempo_onset, tempo_beat_alt, tempo_fine]
        
        # Weighted averaging and median filtering
        bpm_median = np.median(bpm_estimates)
        bpm_weighted_avg = (0.4 * tempo_beat + 0.3 * tempo_onset + 0.2 * tempo_beat_alt + 0.1 * tempo_fine)

        # Final BPM estimate (median of median and weighted average)
        bpm_final = round(np.median([bpm_median, bpm_weighted_avg]), 2)

        # Filter unreasonable values to detect reliable BPM range
        if not (40 <= bpm_final <= 200):
            bpm_final = "BPM detection failed or is unreliable"

        key, confidence = detect_key(y_harmonic, sr)
        print("Detected Key:", key)
        print("Confidence Score:", confidence)
            
    except Exception as e:
        # Log the specific error details for debugging
        print("Error analyzing file:", str(e))
        os.remove(file_path)
        return jsonify({'error': 'Error analyzing file', 'details': str(e)}), 500
    
    os.remove(file_path)
    return jsonify({'message': 'File uploaded successfully', 'bpm': bpm_final, 'key': key}), 200


# Run the app
if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True, host="0.0.0.0", port=5000)