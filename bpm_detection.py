import librosa  # Library for audio analysis
import numpy as np  # Library for numerical operations

def to_scalar(value):
    """Converts a value to a scalar float, handling arrays/lists safely."""
    if isinstance(value, (np.ndarray, list)):  # Check if value is a numpy array or list
        return float(value[0]) if len(value) > 0 else 0.0  # Return first element as float if non-empty, else 0.0
    return float(value)  # Convert directly to float if not a list/array

def detect_bpm(y, sr):
    try:
        print("Starting BPM detection.")  # Log the start of BPM detection
        
        # Step 1: Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)  # Remove leading/trailing silence from the audio
        print("Silence trimmed.")  # Log progress
        
        # Step 2: Resample audio for consistency
        target_sr = 22050  # Define standard sampling rate
        if sr != target_sr:  # Check if resampling is needed
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)  # Resample audio to target_sr
            sr = target_sr  # Update sampling rate variable
            print(f"Resampled audio to {target_sr} Hz.")  # Log resampling
        
        # Step 3: Focus on the middle portion of the track
        middle_start = len(y) // 4  # Define start of middle portion
        middle_end = 3 * len(y) // 4  # Define end of middle portion
        y = y[middle_start:middle_end]  # Slice to keep only the middle portion
        print("Middle section of the audio selected.")  # Log selection
        
        # Step 4: Isolate percussive component
        y_harmonic, y_percussive = librosa.effects.hpss(y)  # Separate harmonic and percussive components
        print("Percussive component isolated.")  # Log separation
        
        # Step 5: Calculate onset envelope
        hop_length = 512  # Define hop length for analysis
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)  # Compute onset strength envelope
        print("Onset envelope computed.")  # Log calculation
        
        # Step 6: Autocorrelation for BPM estimation
        autocorr = librosa.autocorrelate(onset_env, max_size=sr // hop_length)  # Compute autocorrelation of onset envelope
        bpm_min, bpm_max = 40, 200  # Define plausible BPM range
        
        # Step 7: Map autocorrelation peaks to BPM
        times = librosa.frames_to_time(np.arange(len(autocorr)), sr=sr, hop_length=hop_length)  # Convert frame indices to time
        bpms = 60 / times  # Convert time intervals to BPM
        valid = (bpms >= bpm_min) & (bpms <= bpm_max)  # Create mask for valid BPM range
        bpms = bpms[valid]  # Filter BPMs within valid range
        autocorr = autocorr[valid]  # Filter corresponding autocorrelation values
        bpm_autocorr = bpms[np.argmax(autocorr)]  # Select BPM corresponding to max autocorrelation peak
        print(f"Primary BPM estimate from autocorrelation: {bpm_autocorr}")  # Log estimate
        
        # Step 8: Tempo estimation using tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)  # Compute tempogram
        ac_global = np.mean(tempogram, axis=1)  # Average tempogram over time
        tempo_lags = librosa.tempo_frequencies(len(ac_global), sr=sr, hop_length=hop_length)  # Convert lags to BPM
        valid_tempo = (tempo_lags >= bpm_min) & (tempo_lags <= bpm_max)  # Filter valid tempo range
        tempo_lags = tempo_lags[valid_tempo]  # Keep valid tempo lags
        ac_global = ac_global[valid_tempo]  # Keep corresponding strength values
        bpm_tempogram = tempo_lags[np.argmax(ac_global)]  # Select BPM with highest strength
        print(f"Secondary BPM estimate from tempogram: {bpm_tempogram}")  # Log estimate
        
        # Step 9: Beat tracking
        tempo_beat, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, hop_length=hop_length)  # Estimate tempo using beat tracking
        tempo_beat = to_scalar(tempo_beat)  # Convert tempo to scalar value
        print(f"Tertiary BPM estimate from beat tracking: {tempo_beat}")  # Log estimate
        
        # Step 10: Aggregate BPM estimates
        bpm_estimates = [bpm_autocorr, bpm_tempogram, tempo_beat]  # Collect all BPM estimates
        bpm_final = round(float(np.median(bpm_estimates)))  # Use median to determine final BPM
        print(f"Final BPM estimate: {bpm_final}")  # Log final BPM
        
        if not (bpm_min <= bpm_final <= bpm_max):  # Validate final BPM
            print("BPM validation failed.")  # Log validation failure
            return "BPM detection failed or is unreliable"  # Return failure message
        
        print("BPM detection completed successfully.")  # Log success
        return bpm_final  # Return final BPM result
    
    except Exception as e:
        print("Error in BPM detection:", str(e))  # Print exception message
        return "BPM detection failed or is unreliable"  # Return failure message
