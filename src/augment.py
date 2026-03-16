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