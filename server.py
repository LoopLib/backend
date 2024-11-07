# Description: This file contains the code for the server that will be used to upload the audio files to the server.
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import librosa  # Import librosa for BPM detection
from pydub import AudioSegment
import numpy as np
from key_detection import detect_key
from pydub.generators import Sine

# Import CORS from flask_cors
app = Flask(__name__)
# Add CORS to the app
CORS(app, resources={r"/upload": {"origins": "*"}})  # Allow all origins for /upload
# Set the upload folder
app.config['UPLOAD_FOLDER'] = './uploads'
# Create the upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function to safely extract scalar from array or sequence
# In order to handle numpy arrays and other sequence types
def to_scalar(value):
    if isinstance(value, np.ndarray):
        # Convert numpy array to float
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
        # Convert .mp3 to .wav 
        if file_path.endswith(".mp3"):
            audio = AudioSegment.from_mp3(file_path)
            file_path_wav = file_path.replace(".mp3", ".wav")
            audio.export(file_path_wav, format="wav")
            y, sr = librosa.load(file_path_wav)
            os.remove(file_path_wav)
        else:
            y, sr = librosa.load(file_path)

        print("File loaded successfully. Starting BPM detection.")

        # Add metronome track at an estimated BPM, for example, 120 BPM
        bpm_estimate = 120  # This is a placeholder. Set the intended or estimated BPM here.
        duration = librosa.get_duration(y=y, sr=sr)
        y = add_metronome(y, sr, bpm_estimate, duration)

        # Trim silence from start and end and normalize volume
        # Trim at 20 dB from max value, keeping the middle 50% of the signal
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Take a stable segment for consistent BPM
        # Take the middle half of the signal to avoid intro/outro variations
        middle_start = len(y) // 4
        middle_end = 3 * len(y) // 4
        # Extract the middle half of the signal
        y = y[middle_start:middle_end]

        # Harmonic-Percussive Separation to isolate percussive elements
        # y_harmonic: harmonic component of the audio signal
        # y_percussive: percussive component of the audio signal 
        # How is working: The harmonic component contains the melodic content, while the percussive component contains the rhythmic content
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

        # Primary tempo detection using beat_track 
        # tempo_beat: estimated tempo in beats per minute (BPM)
        # How is working: The start_bpm parameter is set to 90 to detect faster tempos
        tempo_beat, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=onset_env, start_bpm=90, tightness=100)
        # Convert tempo to scalar value
        tempo_beat = to_scalar(tempo_beat)

        # Secondary tempo detection using librosa.feature.rhythm.tempo
        # tempo_onset: estimated tempo using onset envelope
        # How is working: The tempo is estimated using the onset envelope
        tempos = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
        tempo_onset = to_scalar(np.median(tempos))

        # Tertiary: Alternative parameters to capture other rhythm patterns
        # How is working: The start_bpm parameter is set to 60 to detect slower tempos
        tempo_beat_alt, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=onset_env, start_bpm=60, tightness=80)
        tempo_beat_alt = to_scalar(tempo_beat_alt)

        # Fine detection using smaller window size for high temporal resolution
        # How is working: The hop_length parameter is set to 256 to use a smaller window size
        hop_length = 256  # smaller window
        # Compute onset strength using the percussive signal
        onset_env_fine = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)
        # Estimate tempo using the onset envelope with a smaller window size
        tempos_fine = librosa.feature.rhythm.tempo(onset_envelope=onset_env_fine, sr=sr, hop_length=hop_length)
        # Compute the median tempo from the fine detection
        tempo_fine = to_scalar(np.median(tempos_fine))

        # Aggregate all detected tempos with weighted voting
        bpm_estimates = [tempo_beat, tempo_onset, tempo_beat_alt, tempo_fine]
        
        # Weighted averaging and median filtering
        # bpm_median: median of all BPM estimates
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

def add_metronome(y, sr, bpm, duration):
    # Generate a metronome click sound (e.g., using a sine wave for simplicity)
    click_freq = 1000  # 1 kHz for a clear percussive click
    click = Sine(click_freq).to_audio_segment(duration=50)  # 50 ms duration for the click
    silence_duration = 60000 / bpm  # Silence between clicks to match BPM
    metronome = click + AudioSegment.silent(duration=silence_duration - 50)
    
    # Create metronome track for the entire audio duration
    metronome_track = metronome * int(duration / (silence_duration / 1000))

    # Convert audio arrays to AudioSegments for overlay
    audio_segment = AudioSegment(
        y.tobytes(),
        frame_rate=sr,
        sample_width=y.dtype.itemsize,
        channels=1
    )
    
    # Overlay metronome and original audio
    combined_audio = audio_segment.overlay(metronome_track)
    
    # Convert back to numpy for librosa processing
    y_combined = np.array(combined_audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return y_combined
# Run the app
if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True, host="0.0.0.0", port=5000)