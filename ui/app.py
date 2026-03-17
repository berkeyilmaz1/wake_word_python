import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.predict import WakeWordDetector

class App:
    def __init__(self, root):
        self.root     = root
        self.detector = None
        self.running  = False

        root.title("Hey Pakize — Wake Word Detector")
        root.geometry("500x420")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_label = tk.Label(
            root, text="⏹  Bekliyor",
            font=("Helvetica", 18, "bold"), fg="gray"
        )
        self.status_label.pack(pady=(30, 10))

        self.confidence_var = tk.DoubleVar(value=0)
        self.progress = tk.Scale(
            root, variable=self.confidence_var,
            from_=0, to=100, orient="horizontal",
            length=400, state="disabled",
            label="Güven (%)"
        )
        self.progress.pack(pady=(0, 20))

        self.btn = tk.Button(
            root, text="🎙  Dinlemeye Başla",
            font=("Helvetica", 13),
            bg="#4CAF50", fg="white",
            width=22, height=2,
            command=self.toggle
        )
        self.btn.pack(pady=(0, 20))

        self.log = scrolledtext.ScrolledText(
            root, height=8, width=58,
            font=("Courier", 10), state="disabled"
        )
        self.log.pack(padx=10)

        self._log("Uygulama başlatıldı. Model yükleniyor...")
        try:
            self.detector = WakeWordDetector(on_result=self.on_result)
            self._log("✅ Model yüklendi. Dinlemeye hazır.")
        except FileNotFoundError as e:
            self._log(f"❌ {e}")
            self.btn.config(state="disabled")

    def toggle(self):
        if not self.running:
            self._start()
        else:
            self._stop()

    def _start(self):
        try:
            self.detector.start()
            self.running = True
            self.btn.config(text="⏹  Durdur", bg="#f44336")
            self.status_label.config(text="🎙  Dinleniyor...", fg="blue")
            self._log("── Dinleme başladı ──────────────────────")
        except Exception as e:
            self._log(f"❌ Mikrofon başlatılamadı: {e}")

    def _stop(self):
        self.detector.stop()
        self.running = False
        self.btn.config(text="🎙  Dinlemeye Başla", bg="#4CAF50")
        self.status_label.config(text="⏹  Bekliyor", fg="gray")
        self.confidence_var.set(0)
        self._log("── Dinleme durduruldu ───────────────────")

    def on_result(self, label, confidence):
        pct = round(confidence * 100, 1)
        self.root.after(0, self._update_ui, label, pct)

    def _update_ui(self, label, pct):
        self.confidence_var.set(pct)
        if label == 1:
            self.status_label.config(text="🟢  HEY PAKİZE!", fg="green")
            ts = datetime.now().strftime("%H:%M:%S")
            self._log(f"[{ts}] 🟢 HEY PAKİZE! — güven: %{pct}")
            self.root.after(2000, self._reset_status)
        else:
            self.status_label.config(text="🎙  Dinleniyor...", fg="blue")

    def _reset_status(self):
        if self.running:
            self.status_label.config(text="🎙  Dinleniyor...", fg="blue")

    def _on_close(self):
        if self.running:
            self.detector.stop()
        self.root.destroy()

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

def run():
    root = tk.Tk()
    App(root)
    root.mainloop()