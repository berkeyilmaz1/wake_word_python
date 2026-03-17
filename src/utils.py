import config  as CONFIG
import numpy as np
import librosa

def standardize(signal):
    """Padding veya cropping ile sabit uzunluğa getirir."""
    target = int(CONFIG.SAMPLE_RATE * CONFIG.TARGET_DURATION)
    if len(signal) < target:
        signal = np.pad(signal, (0, target - len(signal)))
    else:
        signal = signal[:target]
    return signal

def extract_mfcc(y, sr):
    y = standardize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
    
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mean = np.mean(mfcc,    axis=1)
    std  = np.std(mfcc,     axis=1)
    d1   = np.mean(delta,   axis=1)
    d2   = np.mean(delta2,  axis=1)

    return np.concatenate([mean, std, d1, d2])