# Import required libraries
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and MultiIndex handling
import librosa  # For audio processing and feature extraction
from scipy import stats  # For statistical computations
from collections import OrderedDict  # To maintain the order of statistics in feature calculations

# Reference: https://github.com/mdeff/fma/blob/master/features.py

# Create a MultiIndex for audio features with feature type, statistic, and number
def fma_feature_columns():
    feature_sizes = dict(
        chroma_stft=12, chroma_cqt=12, chroma_cens=12,
        tonnetz=6, mfcc=20, rmse=1, zcr=1,
        spectral_centroid=1, spectral_bandwidth=1,
        spectral_contrast=7, spectral_rolloff=1
    )  # Define the number of components for each feature type

    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')  # Statistics to compute
    columns = []  # Initialize list to hold column tuples
    for name, size in feature_sizes.items():  # Loop through feature types and sizes
        for moment in moments:  # Loop through each statistical moment
            for i in range(size):  # Loop through each component
                columns.append((name, moment, f"{i+1:02d}"))  # Create tuple and append
    return pd.MultiIndex.from_tuples(columns, names=["feature", "statistic", "number"]).sort_values()
    # Return a sorted MultiIndex with feature type, statistic, and component number

# Extract features from a single audio file
def compute_features_from_file(file_path):
    feature_index = fma_feature_columns()  # Get predefined feature columns
    features = pd.Series(index=feature_index, dtype=np.float32)  # Initialize Series with all features set to NaN

    # Compute statistical values for a feature array
    def feature_stats(values):
        return OrderedDict([  # Use OrderedDict to maintain order
            ("mean", np.mean(values, axis=1)),  # Mean across time axis
            ("std", np.std(values, axis=1)),  # Standard deviation
            ("skew", stats.skew(values, axis=1)),  # Skewness
            ("kurtosis", stats.kurtosis(values, axis=1)),  # Kurtosis
            ("median", np.median(values, axis=1)),  # Median
            ("min", np.min(values, axis=1)),  # Minimum
            ("max", np.max(values, axis=1)),  # Maximum
        ])

    try:
        x, sr = librosa.load(file_path, sr=None, mono=True)  # Load audio file in mono with original sampling rate
        if np.max(np.abs(x)) > 0:  # Prevent division by zero
            x = x / np.max(np.abs(x))  # Normalize waveform to [-1, 1]
        x = librosa.util.normalize(x)  # Further normalization by peak amplitude

        # Zero-Crossing Rate
        f = librosa.feature.zero_crossing_rate(x)
        for stat_name, stat_values in feature_stats(f).items():  # Compute statistics
            for i, val in enumerate(stat_values):  # Loop through values
                features[("zcr", stat_name, f"{i+1:02d}")] = val  # Assign to Series

        # Constant-Q Transform (CQT)-based features
        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12))  # Compute CQT
        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12)  # Chroma features from CQT
        for stat_name, stat_values in feature_stats(f).items():
            for i, val in enumerate(stat_values):
                features[("chroma_cqt", stat_name, f"{i+1:02d}")] = val

        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12)  # Chroma Energy Normalized Stats (CENS)
        for stat_name, stat_values in feature_stats(f).items():
            for i, val in enumerate(stat_values):
                features[("chroma_cens", stat_name, f"{i+1:02d}")] = val

        f = librosa.feature.tonnetz(chroma=f)  # Tonnetz (tonal centroid) features
        for stat_name, stat_values in feature_stats(f).items():
            for i, val in enumerate(stat_values):
                features[("tonnetz", stat_name, f"{i+1:02d}")] = val

        # Short-Time Fourier Transform (STFT)-based features
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))  # Compute magnitude spectrogram

        f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)  # Chroma from STFT
        for stat_name, stat_values in feature_stats(f).items():
            for i, val in enumerate(stat_values):
                features[("chroma_stft", stat_name, f"{i+1:02d}")] = val

        f = librosa.feature.rms(S=stft)  # Root Mean Square Energy (RMSE)
        for stat_name, stat_values in feature_stats(f).items():
            for i, val in enumerate(stat_values):
                features[("rmse", stat_name, f"{i+1:02d}")] = val

        f = librosa.feature.spectral_centroid(S=stft)  # Spectral centroid
        for stat_name, stat_values in feature_stats(f).items():
            for i, val in enumerate(stat_values):
                features[("spectral_centroid", stat_name, f"{i+1:02d}")] = val

        f = librosa.feature.spectral_bandwidth(S=stft)  # Spectral bandwidth
        for stat_name, stat_values in feature_stats(f).items():
            for i, val in enumerate(stat_values):
                features[("spectral_bandwidth", stat_name, f"{i+1:02d}")] = val

        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)  # Spectral contrast
        for stat_name, stat_values in feature_stats(f).items():
            for i, val in enumerate(stat_values):
                features[("spectral_contrast", stat_name, f"{i+1:02d}")] = val

        f = librosa.feature.spectral_rolloff(S=stft)  # Spectral roll-off
        for stat_name, stat_values in feature_stats(f).items():
            for i, val in enumerate(stat_values):
                features[("spectral_rolloff", stat_name, f"{i+1:02d}")] = val

        # MFCC from mel-spectrogram
        mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)  # Compute mel spectrogram
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)  # Convert to MFCC
        for stat_name, stat_values in feature_stats(f).items():
            for i, val in enumerate(stat_values):
                features[("mfcc", stat_name, f"{i+1:02d}")] = val

        features = features.reindex(feature_index)  # Reorder Series to match MultiIndex

        final_array = features.values.reshape(1, -1).astype(np.float32)  # Convert to 2D array with float32 type

        # print("Final feature shape:", final_array.shape)  # Print shape info
        # print("Final mean/std:", np.mean(final_array), "/", np.std(final_array))  # Print stats of features

        return final_array  # Return the computed feature array

    except Exception as e:
        print("Error extracting features from file:", str(e))  # Print error if feature extraction fails
        return None  # Return None if exception occurs
