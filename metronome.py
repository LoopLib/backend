from pydub import AudioSegment
from pydub.generators import Sine
import numpy as np

def add_metronome(y, sr, bpm, duration):
    click_freq = 1000  # 1 kHz click sound
    click = Sine(click_freq).to_audio_segment(duration=50)
    silence_duration = 60000 / bpm  # Interval between clicks in milliseconds
    metronome = click + AudioSegment.silent(duration=silence_duration - 50)
    metronome_track = metronome * int(duration / (silence_duration / 1000))

    # Convert audio arrays to AudioSegments for overlay
    audio_segment = AudioSegment(
        y.tobytes(),
        frame_rate=sr,
        sample_width=y.dtype.itemsize,
        channels=1
    )

    combined_audio = audio_segment.overlay(metronome_track)
    y_combined = np.array(combined_audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return y_combined
