import librosa
import numpy as np
import tensorflow_hub as hub
import os

# Set TensorFlow Hub cache directory
os.environ["TFHUB_CACHE_DIR"] = "C:/path/to/new/cache_directory"

# Load YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load the full class map from YAMNet (column 2 contains display names)
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
all_classes = np.genfromtxt(class_map_path, dtype=str, delimiter=',', skip_header=1, usecols=2)

# Expanded instrument list
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

def classify_audio(file_path):
    """
    Classifies an audio file using YAMNet. If a vocal/singing class is detected among
    the top predictions, it outputs "Vocals". Otherwise, it checks for instruments.
    """
    y, sr = load_audio(file_path)
    
    # Run YAMNet model (scores shape: [time_steps, 521])
    scores, embeddings, spectrogram = yamnet_model(y)
    print(f"Scores shape: {scores.shape}")
    
    # Average scores across time to get one prediction vector (shape: [521])
    mean_scores = np.mean(scores.numpy(), axis=0)
    
    # Get the indices of the top 10 predictions
    top_indices = np.argsort(mean_scores)[-10:][::-1]
    top_classes = [all_classes[i] for i in top_indices]
    print("Raw top predictions:", top_classes)
    
    # Check for vocals or singing: filter predictions that contain "singing", "vocal", or "speech"
    vocal_predictions = [(class_name, mean_scores[i]) 
                         for i, class_name in zip(top_indices, top_classes)
                         if "sing" in class_name.lower() or "vocal" in class_name.lower() or "speech" in class_name.lower()]
    
    if vocal_predictions:
        vocal_predictions.sort(key=lambda x: x[1], reverse=True)
        best_vocal = vocal_predictions[0][0]
        print(f"Recognized vocals: {best_vocal}")
        return "Vocals"
    
    # If no vocals, filter for instrument labels
    detected_instruments = [(class_name, mean_scores[i]) 
                            for i, class_name in zip(top_indices, top_classes)
                            if class_name in instrument_set]
    
    if detected_instruments:
        detected_instruments.sort(key=lambda x: x[1], reverse=True)
        best_instrument = detected_instruments[0][0]
        print(f"Most likely instrument: {best_instrument}")
        return best_instrument
    
    print("No instrument or vocals detected.")
    return "Unknown"
