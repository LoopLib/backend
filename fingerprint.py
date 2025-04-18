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
        # Load the audio file with its original sampling rate
        y, sr = librosa.load(file_path, sr=None)
        
        # Estimate the tempo and detect beat positions in the audio
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Compute chroma features (pitch class profile) from the audio signal
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Synchronize chroma features to the beat frames using median aggregation
        beat_chroma = librosa.util.sync(chroma, beats, aggregate=np.median)
        
        # Initialize list to store key-normalized chroma vectors
        beat_chroma_norm = []

        # Iterate through each beat-synchronized chroma column
        for col in beat_chroma.T:
            # Find the index of the maximum chroma value (assumed tonal center)
            max_idx = np.argmax(col)

            # Circularly shift the vector so the max value is at index 0 (key invariance)
            norm_col = np.roll(col, -max_idx)

            # Append the normalized chroma vector to the list
            beat_chroma_norm.append(norm_col)
        
        # Convert list to NumPy array and transpose to match original shape (12, num_beats)
        beat_chroma_norm = np.array(beat_chroma_norm).T
        
        # Flatten the normalized chroma matrix to create a single fingerprint vector
        fingerprint_vector = beat_chroma_norm.flatten()
        
        # Normalize the fingerprint vector to unit length (L2 norm)
        norm = np.linalg.norm(fingerprint_vector)
        if norm > 0:
            fingerprint_vector = fingerprint_vector / norm
        
        # Generate a SHA-256 hash of the fingerprint vector bytes
        hash_obj = hashlib.sha256(fingerprint_vector.tobytes())

        # Encode the hash digest using Base64 and return as string
        fingerprint = base64.b64encode(hash_obj.digest()).decode('utf-8')
        return fingerprint

    except Exception as e:
        # Raise an error if anything goes wrong during processing
        raise RuntimeError(f"Error generating fingerprint: {e}")
