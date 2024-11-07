import numpy as np
import librosa

# Constants for major and minor key profiles
MAJOR_PROFILE = np.array([
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # C Major
    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C# Major
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # D Major
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # D# Major
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # E Major
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # F Major
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # F# Major
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],  # G Major
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # G# Major
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # A Major
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # A# Major
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]   # B Major
])

MINOR_PROFILE = np.array([
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # C minor
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C# minor
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # D minor
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],  # D# minor
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # E minor
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # F minor
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # F# minor
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # G minor
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # G# minor
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # A minor
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],  # A# minor
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]   # B minor
])

KEY_CLASSIFICATIONS = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
    'Cmin', 'C#min', 'Dmin', 'D#min', 'Emin', 'Fmin', 'F#min', 'Gmin', 'G#min', 'Amin', 'A#min', 'Bmin'
]

# Function to detect the key of an audio signal
# Description: This function detects the key of an audio signal using chroma features
# Input: y - audio signal, sr - sample rate
# Output: key of the audio signal
def detect_key(y, sr):
    # Detect the key of the input audio signal
    try:
        # Separate harmonic and percussive components
        # Harmonic components are the tonal parts of the audio signal
        # Percussive components are the rhythmic parts of the audio signal
        y_harmonic, _ = librosa.effects.hpss(y)
        # Further separate the harmonic component into harmonic and percussive components
        y_harmonic = librosa.effects.percussive(y_harmonic)

        # Compute chroma features from the harmonic signal
        # Chroma features are used to represent the pitch content of the audio signal
        # Chroma is a 12-dimensional vector that represents the 12 different pitch classes
        # Chroma CQT is more robust to noise, but Chroma STFT is more precise
        chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)

        # Smooth the chroma features by taking the median along the time axis
        # This helps to reduce the effect of transient noise in the signal
        # The resulting chroma vectors are more stable and representative of the overall pitch content
        chroma_cqt_smooth = np.median(chroma_cqt, axis=1)
        chroma_stft_smooth = np.median(chroma_stft, axis=1)

        # Combine the smoothed chroma features from CQT and STFT
        # Weighted combination to give more importance to CQT
        combined_chroma = 0.7 * chroma_cqt_smooth + 0.3 * chroma_stft_smooth

        # Normalize the combined chroma vector
        # This ensures that the vector has unit length, which is necessary for cosine similarity
        # Cosine similarity is used to compare the chroma vector with the key profiles
        similarities_major = [np.dot(combined_chroma, kp) / (np.linalg.norm(combined_chroma) * np.linalg.norm(kp)) for kp in MAJOR_PROFILE]
        similarities_minor = [np.dot(combined_chroma, kp) / (np.linalg.norm(combined_chroma) * np.linalg.norm(kp)) for kp in MINOR_PROFILE]

        # Find the key with the highest similarity score
        # If the highest similarity score is for a major key, return the corresponding major key
        max_major = max(similarities_major)
        # If the highest similarity score is for a minor key, return the corresponding minor key
        max_minor = max(similarities_minor)

        # Determine the best key based on the maximum similarity score
        if max_major >= max_minor:
            # If the highest similarity score is for a major key
            best_key_index = np.argmax(similarities_major)
            # Return the corresponding major key
            best_key = KEY_CLASSIFICATIONS[best_key_index]
        else:
            # If the highest similarity score is for a minor key
            best_key_index = np.argmax(similarities_minor)
            # Return the corresponding minor key
            # 12 is added to the index to get the minor key classification
            best_key = KEY_CLASSIFICATIONS[best_key_index + 12]

        return best_key
    except Exception as e:
        print("Error detecting key:", str(e))
        return "Key detection failed"
