# Audio Analysis API

## Overview

This project provides a Flask-based API for comprehensive audio file analysis, including BPM detection, key detection, instrument classification, fingerprint generation, and genre prediction. Powered by [librosa](https://librosa.org/), TensorFlow Hub, and a pretrained Random Forest model, it enables robust extraction of musical features and metadata.

## Features

- **BPM Detection**: Uses onset envelope, autocorrelation, tempogram, and beat tracking to estimate tempo. (`bpm_detection.py`)
- **Key Detection**: Employs Krumhansl–Schmuckler key profiles and chroma CENS features for key classification. (`key_detection.py`)
- **Instrument Classification**: Utilizes the YAMNet model to detect vocals or specific instruments. (`instrument_detection.py`)
- **Fingerprint Generation**: Generates a tempo- and key-invariant audio fingerprint via chroma features and SHA-256 hashing. (`fingerprint.py`)
- **Genre Prediction**: Extracts comprehensive audio features and uses a pretrained Random Forest classifier to predict music genre. (`compute_features.py`, `genre_detection.py`)
- **REST API**: Exposes endpoints for file uploads and segment analysis. (`server.py`)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/audio-analysis-api.git
   cd audio-analysis-api
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare model artifacts**:
   - Place `rf.joblib`, `scaler.joblib`, and `label_encoder.joblib` in the project root for genre prediction.
   - Ensure TensorFlow Hub can access the internet or has cached the YAMNet model.

## Usage

### Running the Server

```bash
export FLASK_APP=server.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
```

### API Endpoints

#### `POST /upload`

Analyze an entire audio file.

- **Form Data**:
  - `file`: Audio file (`.wav`, `.mp3`, etc.)

- **Response**:
  ```json
  {
    "message": "File uploaded successfully",
    "bpm": 120,
    "key": "C#min",
    "instrument": "Piano",
    "fingerprint": "AbcDefGhIjK...",
    "genre": "Jazz",
    "genre_confidence": 85.23
  }
  ```

#### `POST /analyze_segment`

Analyze a raw audio segment.

- **JSON**:
  ```json
  {
    "segment": [0.0, 0.1, ...],  // Float array
    "sr": 44100
  }
  ```

- **Response**:
  ```json
  {
    "key": "G",
    "confidence": 92.45
  }
  ```

## Module Reference

### `bpm_detection.py`

- `detect_bpm(y, sr) -> int or str`
  - Uses librosa to trim silence, isolate percussive component, compute onset envelope, and estimate BPM via multiple methods.

### `compute_features.py`

- `compute_features_from_file(file_path) -> np.ndarray`
  - Extracts spectral, temporal, and tonal features and returns a flattened feature vector.

### `fingerprint.py`

- `generate_fingerprint(file_path) -> str`
  - Creates a Base64-encoded SHA-256 fingerprint invariant to BPM and key.

### `genre_detection.py`

- `predict_genre(file_path) -> (str, float)`
  - Scales extracted features and predicts genre with confidence.

### `instrument_detection.py`

- `classify_audio(file_path) -> str`
  - Classifies audio using YAMNet to detect vocals or instruments.

### `key_detection.py`

- `detect_key(y, sr) -> (str, float)`
  - Detects major/minor key using chroma CENS and correlation with Krumhansl–Schmuckler profiles.

### `server.py`

- `Flask` application exposing analysis endpoints and orchestrating processing modules.

## Requirements

```text
Flask
Flask-Cors
librosa
numpy
pandas
scipy
joblib
tensorflow-hub
pydub
boto3
scikit-learn
```  


