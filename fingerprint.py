import librosa
import numpy as np
import base64
import hashlib

def generate_fingerprint(file_path):
    """
    Generates a robust audio fingerprint that is more invariant to changes in BPM and key.

    This function:
      - Loads the audio and estimates its tempo.
      - Computes chroma features and synchronizes them to the detected beats.
      - Normalizes each beat's chroma vector by circularly shifting it so that its maximum value
        aligns at index 0 (providing key invariance).
      - Flattens the normalized matrix, normalizes the vector, and creates a SHA-256 hash,
        which is then encoded in Base64.

    Returns:
        A Base64-encoded string fingerprint.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Estimate tempo and obtain beat frames
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Compute chroma features (captures pitch class information)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Synchronize the chroma features with the beats using median aggregation
        beat_chroma = librosa.util.sync(chroma, beats, aggregate=np.median)
        
        # Normalize each beat's chroma vector for key invariance:
        # Shift each column so that the maximum value is at index 0.
        beat_chroma_norm = []
        for col in beat_chroma.T:
            max_idx = np.argmax(col)
            norm_col = np.roll(col, -max_idx)
            beat_chroma_norm.append(norm_col)
        beat_chroma_norm = np.array(beat_chroma_norm).T  # Shape: (12, num_beats)
        
        # Flatten the normalized matrix to form a fingerprint vector
        fingerprint_vector = beat_chroma_norm.flatten()
        
        # Normalize the vector to unit length
        norm = np.linalg.norm(fingerprint_vector)
        if norm > 0:
            fingerprint_vector = fingerprint_vector / norm
        
        # Generate a hash from the fingerprint vector using SHA-256
        hash_obj = hashlib.sha256(fingerprint_vector.tobytes())
        fingerprint = base64.b64encode(hash_obj.digest()).decode('utf-8')
        return fingerprint
    except Exception as e:
        raise RuntimeError(f"Error generating fingerprint: {e}")
