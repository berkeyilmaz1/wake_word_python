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

def extract_mfcc(signal, sr):
    """
    Ses verisinden MFCC öznitelik vektörü çıkarır.
    Eğitimde ve runtime'da birebir aynı fonksiyon çağrılır.
    """
    signal = standardize(signal)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=CONFIG.N_MFCC)
    mean = np.mean(mfcc, axis=1)
    std  = np.std(mfcc, axis=1)
    return np.concatenate([mean, std])