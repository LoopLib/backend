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


