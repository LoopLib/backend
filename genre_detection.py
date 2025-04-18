# Import the function to compute features from an audio file
from compute_features import compute_features_from_file

# Import joblib for loading serialized machine learning models and scalers
import joblib

# Import numpy for numerical operations
import numpy as np

# Load the pre-trained Random Forest model from a file
model = joblib.load("rf.joblib")

# Load the pre-fitted scaler used to normalize the features
scaler = joblib.load("scaler.joblib")

# Load the label encoder to convert between genre labels and numeric form
label_enc = joblib.load("label_encoder.joblib")

# Define a function to predict the genre of a song from its file path
def predict_genre(file_path):
    # Extract features from the audio file
    features = compute_features_from_file(file_path)
    
    # Check if feature extraction failed or if the feature shape is incompatible with the scaler
    if features is None or features.shape[1] != scaler.mean_.shape[0]:
        return "Unknown", 0.0  # Return default values if invalid

    # Scale the features using the previously fitted scaler
    features_scaled = scaler.transform(features)

    # Predict the genre using the trained model
    prediction = model.predict(features_scaled)

    # Get the highest predicted probability as the confidence
    confidence = np.max(model.predict_proba(features_scaled))

    # Convert the numeric prediction back to a genre label
    genre = label_enc.inverse_transform(prediction)[0]

    # Print the shape of the extracted features
    # print("Feature shape:", features.shape)

    # Print the expected input shape for the scaler
    # print("Scaler expects:", scaler.mean_.shape)

    # Print statistical properties of the features before scaling
    # print("Feature stats before scaling:")
    # print("Mean:", np.mean(features))  # Mean of raw features
    # print("Std:", np.std(features))    # Standard deviation of raw features
    # print("Min:", np.min(features))    # Minimum value in raw features
    # print("Max:", np.max(features))    # Maximum value in raw features

    # Get the probability distribution over all classes
    proba = model.predict_proba(features_scaled)[0]

    # Print the probability of each genre
    # for label, p in zip(label_enc.classes_, proba):
        # print(f"{label}: {p:.4f}")

    # Return the predicted genre and the associated confidence
    return genre, confidence
