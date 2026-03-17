import os
import sys
import numpy as np
import sounddevice as sd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.utils import extract_mfcc

class WakeWordDetector:
    def __init__(self, on_result):
        self.on_result = on_result
        self.running   = False
        self._stream   = None

        model_path  = os.path.join(config.MODEL_DIR, "svm_model.pkl")
        scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError("Model bulunamadı! Önce train_pipeline.py çalıştırın.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Scaler bulunamadı! Önce train_pipeline.py çalıştırın.")

        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def _callback(self, indata, frames, time, status):
        if not self.running:
            return

        y = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        # Sessizliği filtrele
        rms = np.sqrt(np.mean(y ** 2))
        if rms < 0.01:
            self.on_result(0, 0.0)
            return

        features    = extract_mfcc(y, config.SAMPLE_RATE)
        scaled      = self.scaler.transform([features])
        probability = self.model.predict_proba(scaled)[0][1]  # her zaman pozitif olasılığı

        if probability >= config.WAKE_WORD_THRESHOLD:
            self.on_result(1, probability)
        else:
            self.on_result(0, probability)

    def start(self):
        self.running = True
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(config.SAMPLE_RATE * config.TARGET_DURATION),
            callback=self._callback
        )
        self._stream.start()

    def stop(self):
        self.running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None