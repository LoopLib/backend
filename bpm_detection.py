import librosa
import numpy as np

def to_scalar(value):
    if isinstance(value, (np.ndarray, list)):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)

def detect_bpm(y, sr):
    try:
        print("Starting BPM detection.")

        # Preprocessing
        # Trim leading and trailing silence
        y, _ = librosa.effects.trim(y, top_db=20)
        print("Silence trimmed.")

        # Resample to a common sample rate if necessary
        target_sr = 22050
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            print(f"Resampled audio to {target_sr} Hz.")

        # Focus on the middle section to avoid intros and outros
        middle_start = len(y) // 4
        middle_end = 3 * len(y) // 4
        y = y[middle_start:middle_end]
        print("Middle section of audio selected.")

        # Isolate percussive component
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        print("Percussive component isolated.")

        # Apply band-pass filter to focus on percussive frequencies (150–5000 Hz)
        S = librosa.stft(y_percussive)
        freqs = librosa.fft_frequencies(sr=sr)
        # Create a mask for frequencies between 150 Hz and 5000 Hz
        filter_mask = (freqs >= 150) & (freqs <= 5000)
        S_filtered = S.copy()
        S_filtered[~filter_mask, :] = 0  # Zero out frequencies outside the desired range
        y_percussive_filtered = librosa.istft(S_filtered)
        print("Band-pass filter applied.")

        # Compute onset strength envelope
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y_percussive_filtered, sr=sr, hop_length=hop_length)
        print("Onset envelope computed.")

        # Compute autocorrelation of the onset envelope
        autocorr = librosa.autocorrelate(onset_env, max_size=sr//hop_length)
        print("Autocorrelation computed.")

        # Define BPM range
        bpm_min = 40
        bpm_max = 200
        # Convert lags to BPM
        times = librosa.frames_to_time(np.arange(len(autocorr)), sr=sr, hop_length=hop_length)
        bpms = 60 / times
        # Only consider reasonable BPMs
        valid = (bpms >= bpm_min) & (bpms <= bpm_max)
        bpms = bpms[valid]
        autocorr = autocorr[valid]

        # Find the BPM with the highest autocorrelation peak
        bpm_candidate = bpms[np.argmax(autocorr)]
        print(f"Primary BPM candidate: {bpm_candidate}")

        # Compute tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        # Aggregate tempogram across time to get global tempo estimate
        ac_global = np.mean(tempogram, axis=1)
        # Convert lag indices to BPM
        tempo_lags = librosa.tempo_frequencies(len(ac_global), sr=sr, hop_length=hop_length)
        # Only consider reasonable BPMs
        valid_tempo = (tempo_lags >= bpm_min) & (tempo_lags <= bpm_max)
        tempo_lags = tempo_lags[valid_tempo]
        ac_global = ac_global[valid_tempo]

        # Find the BPM with the highest tempogram peak
        bpm_tempogram = tempo_lags[np.argmax(ac_global)]
        print(f"Secondary BPM candidate from tempogram: {bpm_tempogram}")

        # Use beat_track as an additional estimation
        tempo_beat, _ = librosa.beat.beat_track(y=y_percussive_filtered, sr=sr, hop_length=hop_length, start_bpm=120, tightness=100)
        tempo_beat = to_scalar(tempo_beat)
        print(f"Third BPM candidate from beat_track: {tempo_beat}")

        # Aggregate all BPM estimates
        bpm_estimates = [bpm_candidate, bpm_tempogram, tempo_beat]
        # Use median to reduce the effect of outliers
        bpm_final = round(float(np.median(bpm_estimates)), 2)
        print(f"Final BPM estimate: {bpm_final}")

        # Validate BPM
        if not (bpm_min <= bpm_final <= bpm_max):
            bpm_final = "BPM detection failed or is unreliable"
            print("BPM validation failed.")
        else:
            print("BPM detection completed successfully.")

        return bpm_final
    except Exception as e:
        print("Error in BPM detection:", str(e))
        return "BPM detection failed or is unreliable"
