import librosa
import config  as CONFIG

def load_audio(path):
    signal, sr = librosa.load(path, sr=CONFIG.SAMPLE_RATE)
    return signal, sr