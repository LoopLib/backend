import numpy as np
import librosa
import pickle

# Load the pre-trained genre detection model
MODEL_PATH = './genre_detection_model.pkl'

with open(MODEL_PATH, 'rb') as model_file:
    genre_model = pickle.load(model_file)


