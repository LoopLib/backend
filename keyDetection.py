import numpy as np
import librosa

# Constants for major and minor key profiles (normalized for efficiency)
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
]) / np.linalg.norm(np.ones(12))

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
]) / np.linalg.norm(np.ones(12))

KEY_CLASSIFICATIONS = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
    'Cmin', 'C#min', 'Dmin', 'D#min', 'Emin', 'Fmin', 'F#min', 'Gmin', 'G#min', 'Amin', 'A#min', 'Bmin'
]

# Function to detect the key of an audio signal
def detect_key(y, sr):
    # Detect the key of the input audio signal
    try:
        # Separate harmonic and percussive components
        # y_harmonic: harmonic component of the audio signal
        #_ : percussive component of the audio signal
        y_harmonic, _ = librosa.effects.hpss(y)

        # Perform tuning correction to improve chroma accuracy
        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        print("Estimated tuning:", tuning)  # Debug: check tuning value
        # Shift the pitch of the harmonic signal by the estimated tuning value
        semitone_shift = -tuning * 12
        # Apply pitch shift to the harmonic signal
        # y_harmonic: pitch-shifted harmonic signal
        y_harmonic = librosa.effects.pitch_shift(y_harmonic, sr=sr, n_steps=semitone_shift)

        # Compute chroma features from the harmonic signal
        # Charoma features represent the energy distribution of pitch classes
        # CQT chroma: chroma features from the constant-Q transform
        # STFT chroma: chroma features from the short-time Fourier transform
        chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)

        # Debug: Print chroma shapes to verify correct extraction
        print("Chroma CQT shape:", chroma_cqt.shape)
        print("Chroma STFT shape:", chroma_stft.shape)

        # Measure harmonic stability across frames
        # Harmonic stability is computed as the mean standard deviation of chroma features
        harmonic_stability = np.std(chroma_cqt, axis=1).mean()
        print("Harmonic stability:", harmonic_stability)  # Debug: check stability

        # Adaptive weighting based on harmonic stability
        # Higher stability -> more weight on CQT chroma
        weight_cqt = min(1, max(0, 0.5 + 0.5 * (1 - harmonic_stability)))
        weight_stft = 1 - weight_cqt
        print("Weights - CQT:", weight_cqt, "STFT:", weight_stft)  # Debug: check weights

        # Smooth the chroma features by taking the median along the time axis
        # Median filtering helps to reduce noise and outliers in the chroma features
        chroma_cqt_smooth = np.median(chroma_cqt, axis=1)
        chroma_stft_smooth = np.median(chroma_stft, axis=1)

        # Combine the smoothed chroma features using adaptive weights
        # Combined chroma vector is used for key estimation
        combined_chroma = weight_cqt * chroma_cqt_smooth + weight_stft * chroma_stft_smooth

        # Normalize the combined chroma vector
        # It counts the number of times each pitch class appears in the signal
        similarities_major = [np.dot(combined_chroma, kp) / (np.linalg.norm(combined_chroma) * np.linalg.norm(kp)) for kp in MAJOR_PROFILE]
        similarities_minor = [np.dot(combined_chroma, kp) / (np.linalg.norm(combined_chroma) * np.linalg.norm(kp)) for kp in MINOR_PROFILE]

        # Debug: Check similarity scores
        print("Similarities Major:", similarities_major)
        print("Similarities Minor:", similarities_minor)

        # Find the key with the highest similarity score
        # The key with the highest similarity score is the estimated key
        max_major = max(similarities_major)
        max_minor = max(similarities_minor)

        # Determine the best key based on the maximum similarity score
        if max_major >= max_minor:
            # Major key
            # Find the index of the key with the highest similarity score
            best_key_index = np.argmax(similarities_major)
            # Retrieve the key label based on the index
            best_key = KEY_CLASSIFICATIONS[best_key_index]
            confidence = max_major
        else:
            # Minor key
            best_key_index = np.argmax(similarities_minor)
            # Retrieve the key label based on the index
            best_key = KEY_CLASSIFICATIONS[best_key_index + 12]
            confidence = max_minor

        # Scale confidence score to make it clearer
        confidence = round(confidence * 100, 2)
        
        return best_key, confidence
    except Exception as e:
        print("Error detecting key:", str(e))
        return "Key detection failed", 0.0
