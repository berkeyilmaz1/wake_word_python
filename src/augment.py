import librosa
import config  as CONFIG
import numpy as np
import os
import soundfile as sf 

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

def augment_file(file_path,output_path,n_copies):
    signal, sr = load_audio(file_path)
    signal = standardize(signal)

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    #Her versiyona 1-3 arası rastgele bir augmentasyon uygula
    for i in range(n_copies):
        augmented = signal.copy()
        ops = np.random.choice(
            [add_noise,
             lambda a: change_pitch(a, sr),
             change_speed],
            size=np.random.randint(1, 3),
            replace=False
        )

        for op in ops:
            augmented = op(augmented)
        
        signal_aug= standardize(augmented)
        output_file = os.path.join(output_path, f"{base_name}_aug_{i}.wav")
        sf.write(output_file, signal_aug, sr)