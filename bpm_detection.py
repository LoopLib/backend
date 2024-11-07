
import librosa
import numpy as np

def to_scalar(value):
    if isinstance(value, np.ndarray):
        return float(value[0]) if value.size > 0 else 0.0
    return float(value)

def detect_bpm(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
    
    # Detect primary tempo
    tempo_beat, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=onset_env, start_bpm=90, tightness=100)
    tempo_beat = to_scalar(tempo_beat)

    # Secondary tempo detection
    tempos = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
    tempo_onset = to_scalar(np.median(tempos))

    # Tertiary tempo detection for slower tempos
    tempo_beat_alt, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, onset_envelope=onset_env, start_bpm=60, tightness=80)
    tempo_beat_alt = to_scalar(tempo_beat_alt)

    # Fine detection using smaller window size
    hop_length = 256
    onset_env_fine = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)
    tempos_fine = librosa.feature.rhythm.tempo(onset_envelope=onset_env_fine, sr=sr, hop_length=hop_length)
    tempo_fine = to_scalar(np.median(tempos_fine))

    # Aggregate all detected tempos with weighted voting
    bpm_estimates = [tempo_beat, tempo_onset, tempo_beat_alt, tempo_fine]
    bpm_median = np.median(bpm_estimates)
    bpm_weighted_avg = (0.4 * tempo_beat + 0.3 * tempo_onset + 0.2 * tempo_beat_alt + 0.1 * tempo_fine)
    bpm_final = round(np.median([bpm_median, bpm_weighted_avg]), 2)

    # Ensure BPM is within reasonable range
    return bpm_final if 40 <= bpm_final <= 200 else "BPM detection failed or is unreliable"
