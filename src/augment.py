import librosa
import config  as CONFIG
import numpy as np

def load_audio(path):
    signal, sr = librosa.load(path, sr=CONFIG.SAMPLE_RATE)
    return signal, sr

def standardize(signal):
    """Padding veya cropping ile sabit uzunluğa getirir."""
    target = int(CONFIG.SAMPLE_RATE * CONFIG.TARGET_DURATION)
    if len(signal) < target:
        signal = np.pad(signal, (0, target - len(signal)))
    else:
        signal = signal[:target]
    return signal

def add_noise(signal, noise_factor=CONFIG.NOISE_LEVEL):
    noise = np.random.randn(len(signal)) * noise_factor
    return signal + noise

def change_pitch(signal, sr):
    steps = np.random.uniform(-CONFIG.PITCH_RANGE, CONFIG.PITCH_RANGE)
    return librosa.effects.pitch_shift(signal, sr, n_steps=steps)

def change_speed(signal):
    rate = np.random.uniform(*CONFIG.SPEED_RANGE)
    return librosa.effects.time_stretch(signal, rate)