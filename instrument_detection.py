# Import the librosa library for audio processing
import librosa

# Import NumPy for numerical operations
import numpy as np

# Import TensorFlow Hub for loading pre-trained models
import tensorflow_hub as hub

# Import the os module for environment variable manipulation
import os

# Set the TensorFlow Hub cache directory to a specified local path
os.environ["TFHUB_CACHE_DIR"] = "C:/path/to/new/cache_directory"

# Load the YAMNet pre-trained model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Get the path to the class map file from the YAMNet model
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')

# Load all class names (display names in column 2) from the class map CSV
all_classes = np.genfromtxt(class_map_path, dtype=str, delimiter=',', skip_header=1, usecols=2)

# Define an expanded list of musical instruments to check for
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

# Convert the instrument list into a set for faster lookup operations
instrument_set = set(instrument_list)

def load_audio(file_path, target_sr=16000):
    """Loads an audio file and resamples it to 16kHz mono."""
    # Load the audio file with a target sample rate and mono channel
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # Print debug information about the loaded audio
    print(f"Loaded audio: {file_path}, Sample Rate: {sr}, Audio Shape: {y.shape}")
    
    # Return the audio waveform and sample rate
    return y, sr

def classify_audio(file_path):
    """
    Classifies an audio file using YAMNet. If a vocal/singing class is detected among
    the top predictions, it outputs "Vocals". Otherwise, it checks for instruments.
    """
    # Load and preprocess the audio file
    y, sr = load_audio(file_path)
    
    # Run the audio through the YAMNet model to get prediction scores, embeddings, and spectrogram
    scores, embeddings, spectrogram = yamnet_model(y)
    
    # Print the shape of the scores array for debugging
    print(f"Scores shape: {scores.shape}")
    
    # Average prediction scores across time steps to get a single prediction vector
    mean_scores = np.mean(scores.numpy(), axis=0)
    
    # Get the indices of the top 10 highest scoring classes
    top_indices = np.argsort(mean_scores)[-10:][::-1]
    
    # Retrieve the class names corresponding to the top indices
    top_classes = [all_classes[i] for i in top_indices]
    
    # Print raw top predicted class names
    print("Raw top predictions:", top_classes)
    
    # Filter the top classes for any vocal-related terms
    vocal_predictions = [(class_name, mean_scores[i]) 
                         for i, class_name in zip(top_indices, top_classes)
                         if "sing" in class_name.lower() or "vocal" in class_name.lower() or "speech" in class_name.lower()]
    
    # If any vocal-related class is detected
    if vocal_predictions:
        # Sort the vocal predictions by score in descending order
        vocal_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get the best-matching vocal class
        best_vocal = vocal_predictions[0][0]
        
        # Print the detected vocal class
        print(f"Recognized vocals: {best_vocal}")
        
        # Return "Vocals" as the classification result
        return "Vocals"
    
    # If no vocals detected, filter for instrument-related classes
    detected_instruments = [(class_name, mean_scores[i]) 
                            for i, class_name in zip(top_indices, top_classes)
                            if class_name in instrument_set]
    
    # If any instruments are detected
    if detected_instruments:
        # Sort the instruments by score in descending order
        detected_instruments.sort(key=lambda x: x[1], reverse=True)
        
        # Get the most likely instrument class
        best_instrument = detected_instruments[0][0]
        
        # Print the most likely detected instrument
        print(f"Most likely instrument: {best_instrument}")
        
        # Return the instrument name as the classification result
        return best_instrument
    
    # If neither vocals nor instruments are detected, print fallback message
    print("No instrument or vocals detected.")
    
    # Return "Unknown" as the classification result
    return "Unknown"
