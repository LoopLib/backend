import numpy as np
import librosa

# Reference: https://librosa.org/doc/latest/generated/librosa.key_to_notes.html

# Krumhansl-Schmuckler key profiles for major and minor keys used in order to improve key detection
MAJOR_PROFILE_BASE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE_BASE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Normalize the profiles in order to compare them with chroma features
# The chroma features are normalized, so the key profiles should be too
MAJOR_PROFILE_BASE /= np.linalg.norm(MAJOR_PROFILE_BASE)
MINOR_PROFILE_BASE /= np.linalg.norm(MINOR_PROFILE_BASE)

# Generate profiles for all 12 major and minor keys by rotating the base profiles
# The key profiles are circular, so need to consider all rotations
MAJOR_PROFILES = np.array([np.roll(MAJOR_PROFILE_BASE, i) for i in range(12)])
MINOR_PROFILES = np.array([np.roll(MINOR_PROFILE_BASE, i) for i in range(12)])

# List of key classifications
KEY_CLASSIFICATIONS = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
    'Cmin', 'C#min', 'Dmin', 'D#min', 'Emin', 'Fmin', 'F#min', 'Gmin', 'G#min', 'Amin', 'A#min', 'Bmin'
]

# Function to detect the key of a given audio signal
# param y: audio time series
# param sr: sample rate of the audio signal
# return: detected key and confidence score
# design decision: 
#   1. use the Krumhansl-Schmuckler key profiles for key detection
#   2. normalize the chroma features and key profiles for comparison
#   3. compute the dot product of the chroma vector with the key profiles
#   4. select the key with the highest similarity score
#   5. return the detected key and confidence score
def detect_key(y, sr):
    try:
        # Estimate tuning to adjust the chroma features
        # Tuning is done in order to align the chroma features with the key profiles
        # The tuning value is in cents (100 cents = 1 semitone)
        # y=y is the audio time series, sr=sr is the sample rate
        # Reference: https://librosa.org/doc/main/generated/librosa.pitch_tuning.html
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        # print("Estimated tuning:", tuning)  # Debug: check tuning value

        # Compute energy-normalized chroma features (Chroma CENS)
        # Chroma features are used to represent the harmonic content of the audio signal
        # Chroma CENS provides robustness to noise and variations in timbre (instrumentation)
        # CENS: https://librosa.org/doc/main/generated/librosa.feature.chroma_cens.html
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, tuning=tuning)
        # print("Chroma CENS shape:", chroma.shape)  # Debug: check chroma shape

        # Average chroma over time to get a single chroma vector
        # To summarize the harmonic content of the entire audio signal
        # axis=1 means averaging along the time axis
        chroma_mean = np.mean(chroma, axis=1)
        # print("Chroma mean:", chroma_mean)  # Debug: check chroma mean

        # Normalize the chroma vector to compare the chroma vector with the key profiles
        # Reference: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        chroma_norm = chroma_mean / np.linalg.norm(chroma_mean)
        # print("Normalized chroma vector:", chroma_norm)  # Debug: check chroma normalization

        # Compute similarities with major and minor profiles using dot product
        # Purpose is to find the key with the highest similarity score
        # .dot() computes the dot product of two arrays
        # dot product = sum of the element-wise products
        similarities_major = np.dot(MAJOR_PROFILES, chroma_norm)
        similarities_minor = np.dot(MINOR_PROFILES, chroma_norm)
        # print("Similarities Major:", similarities_major)
        # print("Similarities Minor:", similarities_minor)

        # Select the key with the highest similarity score
        max_major = np.max(similarities_major)
        max_minor = np.max(similarities_minor)

        # Determine the best key and confidence score
        # If the major key has a higher similarity score, select the major key
        # Otherwise, select the minor key
        # np.argmax() returns the index of the maximum value in an array
        if max_major >= max_minor:
            best_key_index = np.argmax(similarities_major)
            best_key = KEY_CLASSIFICATIONS[best_key_index]
            confidence = max_major
        else:
            best_key_index = np.argmax(similarities_minor)
            best_key = KEY_CLASSIFICATIONS[best_key_index + 12]
            confidence = max_minor

        # Scale confidence score to percentage (0-100)
        confidence = round(float(confidence) * 100, 2)

        return best_key, confidence
    except Exception as e:
        print("Error detecting key:", str(e))
        return "Key detection failed", 0.0
