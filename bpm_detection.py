import librosa
import numpy as np

def to_scalar(value):
    """Converts a value to a scalar float, handling arrays/lists safely."""
    if isinstance(value, (np.ndarray, list)):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)

def detect_bpm(y, sr):
    try:
        print("Starting BPM detection.")
        
        # Step 1: Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        print("Silence trimmed.")
        
        # Step 2: Resample audio for consistency
        target_sr = 22050
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            print(f"Resampled audio to {target_sr} Hz.")
        
        # Step 3: Focus on the middle portion of the track
        middle_start = len(y) // 4
        middle_end = 3 * len(y) // 4
        y = y[middle_start:middle_end]
        print("Middle section of the audio selected.")
        
        # Step 4: Isolate percussive component
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        print("Percussive component isolated.")
        
        # Step 5: Calculate onset envelope
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)
        print("Onset envelope computed.")
        
        # Step 6: Autocorrelation for BPM estimation
        autocorr = librosa.autocorrelate(onset_env, max_size=sr // hop_length)
        bpm_min, bpm_max = 40, 200
        
        # Step 7: Map autocorrelation peaks to BPM
        times = librosa.frames_to_time(np.arange(len(autocorr)), sr=sr, hop_length=hop_length)
        bpms = 60 / times
        valid = (bpms >= bpm_min) & (bpms <= bpm_max)
        bpms = bpms[valid]
        autocorr = autocorr[valid]
        
        bpm_autocorr = bpms[np.argmax(autocorr)]
        print(f"Primary BPM estimate from autocorrelation: {bpm_autocorr}")
        
        # Step 8: Tempo estimation using tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        ac_global = np.mean(tempogram, axis=1)
        tempo_lags = librosa.tempo_frequencies(len(ac_global), sr=sr, hop_length=hop_length)
        valid_tempo = (tempo_lags >= bpm_min) & (tempo_lags <= bpm_max)
        tempo_lags = tempo_lags[valid_tempo]
        ac_global = ac_global[valid_tempo]
        bpm_tempogram = tempo_lags[np.argmax(ac_global)]
        print(f"Secondary BPM estimate from tempogram: {bpm_tempogram}")
        
        # Step 9: Beat tracking
        tempo_beat, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, hop_length=hop_length)
        tempo_beat = to_scalar(tempo_beat)
        print(f"Tertiary BPM estimate from beat tracking: {tempo_beat}")
        
        # Step 10: Aggregate BPM estimates
        bpm_estimates = [bpm_autocorr, bpm_tempogram, tempo_beat]
        bpm_final = round(float(np.median(bpm_estimates)), 2)
        print(f"Final BPM estimate: {bpm_final}")
        
        if not (bpm_min <= bpm_final <= bpm_max):
            print("BPM validation failed.")
            return "BPM detection failed or is unreliable"
        
        print("BPM detection completed successfully.")
        return bpm_final
    
    except Exception as e:
        print("Error in BPM detection:", str(e))
        return "BPM detection failed or is unreliable"
