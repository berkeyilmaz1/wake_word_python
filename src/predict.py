# src/predict.py
# Mikrofon tahmin motoru.
# Doğrudan çalışmaz — UI tarafından Thread içinde çağrılır.

import os
import sys
import numpy as np
import sounddevice as sd
import joblib
import librosa
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.utils import extract_mfcc

class WakeWordDetector:
    """
    UI'dan şu şekilde kullanılır:
        detector = WakeWordDetector(on_result=callback_fn)
        detector.start()   # dinlemeye başla
        detector.stop()    # durdur
    """

    def __init__(self, on_result):
        """
        on_result: her tahmin sonucunda çağrılacak fonksiyon.
                   (label: int, confidence: float) alır.
        """
        self.on_result  = on_result
        self.running    = False
        self._stream    = None

        model_path  = os.path.join(config.MODEL_DIR, "svm_model.pkl")
        scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model bulunamadı! Önce train_pipeline.py çalıştırın."
            )

        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def _callback(self, indata, frames, time, status):
        if not self.running:
            return
        y = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        features = extract_mfcc(y, config.SAMPLE_RATE)
        features_scaled = self.scaler.transform([features])
        label      = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0][label]
        self.on_result(label, confidence)

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