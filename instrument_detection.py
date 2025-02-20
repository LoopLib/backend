import librosa
import numpy as np
import tensorflow_hub as hub

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
    print(f"✅ Loaded audio: {file_path}, Sample Rate: {sr}, Audio Shape: {y.shape}")
    return y, sr

def classify_audio(file_path):
    """Classifies an audio file using YAMNet and returns the most likely instrument
    (based on our expanded instrument list)."""
    y, sr = load_audio(file_path)
    
    # Run YAMNet model (scores shape: [time_steps, 521])
    scores, embeddings, spectrogram = yamnet_model(y)
    print(f"📊 Scores shape: {scores.shape}")
    
    # Average scores across time to get one prediction vector (shape: [521])
    mean_scores = np.mean(scores.numpy(), axis=0)
    
    # Get the indices of the top 10 predictions
    top_indices = np.argsort(mean_scores)[-10:][::-1]
    top_classes = [all_classes[i] for i in top_indices]
    print("🔍 Raw top predictions:", top_classes)
    
    # Filter for labels that are in our instrument set
    detected_instruments = [(class_name, mean_scores[i]) 
                            for i, class_name in zip(top_indices, top_classes)
                            if class_name in instrument_set]
    
    if detected_instruments:
        # Sort by confidence (score) descending and pick the highest one
        detected_instruments.sort(key=lambda x: x[1], reverse=True)
        best_instrument = detected_instruments[0][0]
        print(f"🎸 Most likely instrument: {best_instrument}")
        return best_instrument
    
    print("❌ No instrument detected.")
    return "Unknown"

