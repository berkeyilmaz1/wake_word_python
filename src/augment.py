import librosa
import config  as CONFIG
import numpy as np
import os
import soundfile as sf 
from src import utils as UTILS

def load_audio(path):
    signal, sr = librosa.load(path, sr=CONFIG.SAMPLE_RATE)
    return signal, sr


def add_noise(signal, noise_factor=CONFIG.NOISE_LEVEL):
    noise = np.random.randn(len(signal)) * noise_factor
    return signal + noise

def change_pitch(signal, sr):
    steps = np.random.uniform(-CONFIG.PITCH_RANGE, CONFIG.PITCH_RANGE)
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=steps)

def change_speed(signal):
    rate = np.random.uniform(*CONFIG.SPEED_RANGE)
    return librosa.effects.time_stretch(signal, rate=rate)

def augment_file(file_path,output_path,n_copies):
    signal, sr = load_audio(file_path)
    signal = UTILS.standardize(signal)

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    #Her versiyona 1-3 arası rastgele bir augmentasyon uygula
    for i in range(n_copies):
        augmented = signal.copy()
        ops = np.random.choice(
            [add_noise,
             lambda a: change_pitch(a, sr)],
            size=np.random.randint(1, 3),
            replace=False
        )

        for op in ops:
            augmented = op(augmented)
        
        signal_aug = UTILS.standardize(augmented)
        output_file = os.path.join(output_path, f"{base_name}_aug_{i}.wav")
        sf.write(output_file, signal_aug, sr)


def run():
    for raw_dir, aug_dir in [
        (CONFIG.RAW_POS_DIR, CONFIG.AUG_POS_DIR),
        (CONFIG.RAW_NEG_DIR, CONFIG.AUG_NEG_DIR)
    ]:
        os.makedirs(aug_dir, exist_ok=True)
        files = [f for f in os.listdir(raw_dir) if f.endswith(".wav")]
        print(f"{raw_dir}: {len(files)} orijinal dosya bulundu")

        for f in files:
            augment_file(
                os.path.join(raw_dir, f),
                aug_dir,
                CONFIG.AUGMENT_COPIES
            )
        print(f"  → {aug_dir} klasörüne yazıldı")

    print("✅ Augmentation tamamlandı.")

if __name__ == "__main__":
    run()