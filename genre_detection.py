import numpy as np
import librosa
import pickle

# Load the pre-trained genre detection model
MODEL_PATH = './genre_detection_model.pkl'

with open(MODEL_PATH, 'rb') as model_file:
    genre_model = pickle.load(model_file)


# Function to extract features for genre detection
def extract_features(y, sr):
    try:
        # Extract MFCCs, chroma, and spectral contrast as features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Aggregate features (mean and standard deviation)
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        features = np.concatenate([mfcc_mean, chroma_mean, spectral_contrast_mean])
        return features
    except Exception as e:
        print("Error extracting features:", str(e))
        return None
    
# Function to predict genre using the pre-trained model
def detect_genre(y, sr):
    try:
        features = extract_features(y, sr)
        if features is None:
            raise ValueError("Feature extraction failed.")

        # Predict genre using the pre-trained model
        genre_prediction = genre_model.predict([features])[0]
        return genre_prediction
    except Exception as e:
        print("Error detecting genre:", str(e))
        return "Unknown"