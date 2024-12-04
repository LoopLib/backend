from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import librosa
from pydub import AudioSegment
import numpy as np
from key_detection import detect_key
from bpm_detection import detect_bpm  # Import the BPM detection function
from genre_detection import detect_genre  # Import the genre detection function
import warnings

# Suppress warnings from librosa
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Log the uploaded file name
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

         # Call the genre detection function
        genre_prediction = detect_genre(y, sr)
        print(f"Predicted Genre: {genre_prediction}")

    except Exception as e:
        # Log the error details
        print("Error analyzing file:", str(e))
        os.remove(file_path)
        return jsonify({'error': 'Error analyzing file', 'details': str(e)}), 500

   # Clean up uploaded file
    os.remove(file_path)
    return jsonify({
        'message': 'File uploaded successfully',
        'bpm': bpm_final,
        'key': key,
        'genre': genre_prediction
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
