from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import librosa
from pydub import AudioSegment
import numpy as np
import warnings
import boto3

from key_detection import detect_key
from bpm_detection import detect_bpm 
from fingerprint import generate_fingerprint

# Suppress warnings from librosa
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Enable CORS for all routes and all origins
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/analyze_segment', methods=['POST'])
def analyze_segment():
    try:
        data = request.json
        segment = np.array(data.get('segment', []))
        sr = data.get('sr', 44100)

        if len(segment) == 0:
            raise ValueError("Received empty audio segment for analysis")

        key, confidence = detect_key(segment, sr)
        print(f"Detected Key: {key}, Confidence: {confidence}%")
        return jsonify({'key': key, 'confidence': confidence}), 200
    except Exception as e:
        print("Error analyzing segment:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    print("Received file:", file.filename)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Convert .mp3 to .wav if necessary
        if file_path.endswith(".mp3"):
            audio = AudioSegment.from_mp3(file_path)
            file_path_wav = file_path.replace(".mp3", ".wav")
            audio.export(file_path_wav, format="wav")
            y, sr = librosa.load(file_path_wav, sr=None)
            os.remove(file_path_wav)
        else:
            y, sr = librosa.load(file_path, sr=None)

        print("File loaded successfully.")

        # Call the BPM detection function
        bpm_final = detect_bpm(y, sr)
        print(f"Detected BPM: {bpm_final}")

        # Isolate harmonic component for key detection
        y_harmonic, _ = librosa.effects.hpss(y)

        # Call the key detection function
        key, confidence = detect_key(y_harmonic, sr)
        print(f"Detected Key: {key} with confidence {confidence}%")

        # Generate fingerprint using your fingerprint module
        fingerprint = generate_fingerprint(file_path)
        print(f"Generated Fingerprint: {fingerprint}")

    except Exception as e:
        print("Error analyzing file:", str(e))
        os.remove(file_path)
        return jsonify({'error': 'Error analyzing file', 'details': str(e)}), 500

    # Clean up uploaded file
    os.remove(file_path)
    return jsonify({
        'message': 'File uploaded successfully',
        'bpm': bpm_final,
        'key': key,
        'fingerprint': fingerprint
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)