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
    mfcc        = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=CONFIG.N_MFCC)
    delta       = librosa.feature.delta(mfcc)
    delta2      = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return np.concatenate([np.mean(features, axis=1), np.std(features, axis=1)])