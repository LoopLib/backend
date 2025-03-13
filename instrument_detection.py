import librosa
import numpy as np
import tensorflow_hub as hub
import os

os.environ["TFHUB_CACHE_DIR"] = "C:/cache_directory"

# Load YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load the full class map from YAMNet (column 2 contains display names)
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
all_classes = np.genfromtxt(class_map_path, dtype=str, delimiter=',', skip_header=1, usecols=2)

# Expanded instrument list to cover many instrument-related labels from YAMNet
instrument_list = [
    "Accordion",
    "Acoustic guitar", "Acoustic guitar (nylon)",
    "Bass drum", "Bass drum (acoustic)",
    "Bass guitar",
    "Cello",
    "Clarinet",
    "Drum", "Drum kit", "Snare drum", "Cymbal",
    "Electric guitar", "Electric guitar (jazz)", "Electric guitar (clean)",
    "Electric piano",
    "Flute",
    "Glockenspiel",
    "Harpsichord",
    "Harmonica",
    "Harp",
    "Mandolin",
    "Marimba",
    "Oboe",
    "Organ",
    "Piano",
    "Saxophone",
    "Synthesizer",
    "Timpani",
    "Trombone",
    "Trumpet",
    "Tuba",
    "Violin",
    "Xylophone",
    "Ukulele",
    "Steel drum"
]

# Convert the list into a set for fast lookup
instrument_set = set(instrument_list)

def load_audio(file_path, target_sr=16000):
    """Loads an audio file and resamples it to 16kHz mono."""
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    print(f"Loaded audio: {file_path}, Sample Rate: {sr}, Audio Shape: {y.shape}")
    return y, sr

def classify_audio(file_path, top_k=20, threshold=0.05):
    """
    Classifies an audio file using YAMNet.
    If a vocal/speech class is detected, returns "Vocals".
    Otherwise, it looks for instrument labels (normalized) among the top predictions.
    If multiple instruments exceed the threshold, returns a list; if only one, returns that string.
    """
    y, sr = load_audio(file_path)
    scores, embeddings, spectrogram = yamnet_model(y)
    mean_scores = np.mean(scores.numpy(), axis=0)
    
    # Check for vocals in all classes (using a threshold)
    vocal_predictions = [
        (all_classes[i].strip(), mean_scores[i])
        for i in range(len(all_classes))
        if "speech" in all_classes[i].lower() and mean_scores[i] > threshold
    ]
    if vocal_predictions:
        vocal_predictions.sort(key=lambda x: x[1], reverse=True)
        best_vocal = vocal_predictions[0][0]
        print(f"Recognized vocals: {best_vocal}")
        return "Vocals"
    
    # Search for instrument predictions in a larger set of top predictions
    top_indices = np.argsort(mean_scores)[-top_k:][::-1]
    detected_instruments = []
    for i in top_indices:
        label = all_classes[i].strip()  # Normalize by stripping whitespace
        if label in instrument_set and mean_scores[i] > threshold:
            detected_instruments.append((label, mean_scores[i]))
    
    if not detected_instruments:
        print("No instrument or vocals detected.")
        return "Unknown"
    
    # Sort detected instruments by score (highest first)
    detected_instruments.sort(key=lambda x: x[1], reverse=True)
    instruments = [inst for inst, score in detected_instruments]
    
    if len(instruments) == 1:
        print(f"Detected instrument: {instruments[0]}")
        return instruments[0]
    else:
        print(f"Detected instruments: {instruments}")
        return instruments


