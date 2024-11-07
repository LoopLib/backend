import numpy as np
import librosa

# Krumhansl-Schmuckler key profiles for major and minor keys
MAJOR_PROFILE_BASE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE_BASE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Normalize the profiles
MAJOR_PROFILE_BASE /= np.linalg.norm(MAJOR_PROFILE_BASE)
MINOR_PROFILE_BASE /= np.linalg.norm(MINOR_PROFILE_BASE)

# Generate profiles for all 12 major and minor keys by rotating the base profiles
MAJOR_PROFILES = np.array([np.roll(MAJOR_PROFILE_BASE, i) for i in range(12)])
MINOR_PROFILES = np.array([np.roll(MINOR_PROFILE_BASE, i) for i in range(12)])

KEY_CLASSIFICATIONS = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
    'Cmin', 'C#min', 'Dmin', 'D#min', 'Emin', 'Fmin', 'F#min', 'Gmin', 'G#min', 'Amin', 'A#min', 'Bmin'
]

def detect_key(y, sr):
    try:
        # Estimate tuning to adjust the chroma features
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        print("Estimated tuning:", tuning)  # Debug: check tuning value

        # Compute energy-normalized chroma features (Chroma CENS)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, tuning=tuning)
        print("Chroma CENS shape:", chroma.shape)  # Debug: check chroma shape

        # Average chroma over time to get a single chroma vector
        chroma_mean = np.mean(chroma, axis=1)
        print("Chroma mean:", chroma_mean)  # Debug: check chroma mean

        # Normalize the chroma vector
        chroma_norm = chroma_mean / np.linalg.norm(chroma_mean)
        print("Normalized chroma vector:", chroma_norm)  # Debug: check chroma normalization

        # Compute similarities with major and minor profiles using dot product
        similarities_major = np.dot(MAJOR_PROFILES, chroma_norm)
        similarities_minor = np.dot(MINOR_PROFILES, chroma_norm)
        print("Similarities Major:", similarities_major)
        print("Similarities Minor:", similarities_minor)

        # Find the best matching key
        max_major = np.max(similarities_major)
        max_minor = np.max(similarities_minor)

        if max_major >= max_minor:
            best_key_index = np.argmax(similarities_major)
            best_key = KEY_CLASSIFICATIONS[best_key_index]
            confidence = max_major
        else:
            best_key_index = np.argmax(similarities_minor)
            best_key = KEY_CLASSIFICATIONS[best_key_index + 12]
            confidence = max_minor

        # Scale confidence score
        confidence = round(float(confidence) * 100, 2)

        return best_key, confidence
    except Exception as e:
        print("Error detecting key:", str(e))
        return "Key detection failed", 0.0
