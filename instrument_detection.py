import librosa
import numpy as np
import tensorflow_hub as hub

# Load YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')


