import librosa
import numpy as np

# Define a helper function to convert a value to a scalar
# To handle the case when the value is an empty list or array
def to_scalar(value):
    # Convert to float if the value is a scalar
    if isinstance(value, (np.ndarray, list)):
        # Return the first element if the array is not empty
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)

# Define the BPM detection function
# This function estimates the tempo of a music audio signal
# param y: audio time series
# param sr: sample rate of the audio signal
# return: estimated BPM (beats per minute) as a string
# design decision:
#   1. preprocess the audio signal to focus on the percussive component
#   2. apply a band-pass filter to isolate percussive frequencies (150–5000 Hz)
#   3. compute the onset strength envelope to detect note onsets
#   4. compute the autocorrelation of the onset envelope to estimate the tempo
#   5. validate the BPM estimate by checking if it falls within a reasonable range
#   6. return the final BPM estimate as a string
def detect_bpm(y, sr):
    try:
        print("Starting BPM detection.")

        # The leading and trailing silence is trimmed to focus on the main audio content
        # top_db=20 means that any audio below -20 dB will be considered as silence
        # Reference: https://librosa.org/doc/0.10.1/generated/librosa.effects.trim.html
        y, _ = librosa.effects.trim(y, top_db=20)
        print("Silence trimmed.")

        # Resample to a common sample rate if necessary
        # Sample rate (sr) is the number of samples per second in the audio signal
        # The target sample rate is 22050 Hz
        # Resampling is done to ensure consistency in the analysis
        # Reference: https://librosa.org/doc/0.10.1/generated/librosa.resample.html
        target_sr = 22050
        if sr != target_sr:
            # Resample the audio signal to the target sample rate
            # orig_sr=sr is the original sample rate, target_sr=target_sr is the target sample rate
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            print(f"Resampled audio to {target_sr} Hz.")

        # Focus on the middle section to avoid intros and outros
        middle_start = len(y) // 4 # Start at 1/4 of the audio
        middle_end = 3 * len(y) // 4 # End at 3/4 of the audio 
        # Extract the middle section of the audio
        y = y[middle_start:middle_end]
        print("Middle section of audio selected.")

        # Isolate percussive component in order to focus on the rhythm
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        print("Percussive component isolated.")

        # Apply band-pass filter to focus on percussive frequencies (150–5000 Hz)
        # Reference: https://librosa.org/doc/0.10.1/generated/librosa.stft.html
        S = librosa.stft(y_percussive)
        # Compute the frequencies corresponding to the STFT
        # Reference: https://librosa.org/doc/0.10.1/generated/librosa.fft_frequencies.html
        freqs = librosa.fft_frequencies(sr=sr)
        # Create a mask for frequencies between 150 Hz and 5000 Hz
        filter_mask = (freqs >= 150) & (freqs <= 5000)
        S_filtered = S.copy()
        S_filtered[~filter_mask, :] = 0  # Zero out frequencies outside the desired range
        y_percussive_filtered = librosa.istft(S_filtered) # Inverse STFT to get the filtered signal
        print("Band-pass filter applied.")

        # Compute onset strength envelope
        # Onset strength is a measure of the energy of note onsets in the signal
        # Reference: https://librosa.org/doc/0.10.1/generated/librosa.onset.onset_strength.html
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y_percussive_filtered, sr=sr, hop_length=hop_length)
        print("Onset envelope computed.")

        # Compute autocorrelation of the onset envelope
        # Autocorrelation is used to find repeating patterns in the onset envelope
        autocorr = librosa.autocorrelate(onset_env, max_size=sr//hop_length)
        print("Autocorrelation computed.")

        # Define BPM range
        bpm_min = 40
        bpm_max = 200
        # Convert lags to BPM
        # Time lag is the time difference between two events
        # lags = time lags in samples, sr = sample rate, hop_length = hop length in samples
        times = librosa.frames_to_time(np.arange(len(autocorr)), sr=sr, hop_length=hop_length)
        bpms = 60 / times

        # Only consider reasonable BPMs
        valid = (bpms >= bpm_min) & (bpms <= bpm_max)
        bpms = bpms[valid]
        autocorr = autocorr[valid]

        # Find the BPM with the highest autocorrelation peak
        bpm_candidate = bpms[np.argmax(autocorr)]
        print(f"Primary BPM candidate: {bpm_candidate}")

        # Compute tempogram to get a secondary BPM estimate
        # Tempogram is a time-frequency representation of tempo
        # Reference: https://librosa.org/doc/0.10.1/generated/librosa.feature.tempogram.html
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        # Aggregate tempogram across time to get global tempo estimate
        # axis=1 means averaging along the time axis
        ac_global = np.mean(tempogram, axis=1)
        # Convert lag indices to BPM
        tempo_lags = librosa.tempo_frequencies(len(ac_global), sr=sr, hop_length=hop_length)
        # Only consider reasonable BPMs
        valid_tempo = (tempo_lags >= bpm_min) & (tempo_lags <= bpm_max)
        tempo_lags = tempo_lags[valid_tempo]
        ac_global = ac_global[valid_tempo]

        # Find the BPM with the highest tempogram peak
        # This serves as a secondary BPM estimate
        bpm_tempogram = tempo_lags[np.argmax(ac_global)]
        print(f"Secondary BPM candidate from tempogram: {bpm_tempogram}")

        # Use beat_track as an additional estimation
        # Beat tracking is a method to estimate the tempo and the beats using a pre-trained model
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
