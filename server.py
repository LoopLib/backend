from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import librosa
from keyDetection import detect_key
from bpm_detection import detect_bpm  # Import detect_bpm from bpm_detection.py
from metronome import add_metronome  # Import add_metronome from metronome.py

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

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        if file_path.endswith(".mp3"):
            audio = AudioSegment.from_mp3(file_path)
            file_path_wav = file_path.replace(".mp3", ".wav")
            audio.export(file_path_wav, format="wav")
            y, sr = librosa.load(file_path_wav)
            os.remove(file_path_wav)
        else:
            y, sr = librosa.load(file_path)

        bpm_estimate = 120
        duration = librosa.get_duration(y=y, sr=sr)
        y = add_metronome(y, sr, bpm_estimate, duration)
        y, _ = librosa.effects.trim(y, top_db=20)
        middle_start = len(y) // 4
        middle_end = 3 * len(y) // 4
        y = y[middle_start:middle_end]

        bpm_final = detect_bpm(y, sr)
        y_harmonic, _ = librosa.effects.hpss(y)
        key, confidence = detect_key(y_harmonic, sr)

    except Exception as e:
        os.remove(file_path)
        return jsonify({'error': 'Error analyzing file', 'details': str(e)}), 500

    os.remove(file_path)
    return jsonify({'message': 'File uploaded successfully', 'bpm': bpm_final, 'key': key}), 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
