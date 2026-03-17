# src/features.py
# Tüm ses dosyalarından MFCC çıkarır → dataset.csv oluşturur.t-SNE ile görselleştirme yapar.
# Sadece train_pipeline.py tarafından çağrılır.

import os
import sys
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.utils import extract_mfcc


def collect(directory, label):
    rows = []
    for f in os.listdir(directory):
        if not f.endswith(".wav"):
            continue
        try:
            y, sr = librosa.load(os.path.join(directory, f), sr=config.SAMPLE_RATE)
            feat = extract_mfcc(y, sr)
            row = {"Dosya_Adi": f}
            row.update({f"F{i}": v for i, v in enumerate(feat)})
            row["Label"] = label
            rows.append(row)
        except Exception as e:
            print(f"  ⚠️ Atlandı: {f} → {e}")
    return rows


def run():
    rows= []
    for d,lbl in [
     (config.RAW_POS_DIR, 1), (config.AUG_POS_DIR, 1),
        (config.RAW_NEG_DIR, 0), (config.AUG_NEG_DIR, 0),
    ]:
        rows += collect(d, lbl)
    df = pd.DataFrame(rows)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(config.OUTPUT_DIR, "dataset.csv")
    df.to_csv(csv_path, index=False)

    pos = (df.Label == 1).sum()
    neg = (df.Label == 0).sum()
    print(f"✅ CSV: {len(df)} örnek | Pozitif: {pos} | Negatif: {neg}")

    if df.isnull().sum().sum() > 0:
        print("⚠️ NaN tespit edildi! Bozuk ses dosyası olabilir.")

    feat_cols = [c for c in df.columns if c.startswith("F")]
    tsne = TSNE(n_components=2, random_state=config.RANDOM_STATE, perplexity=30)
    X2d  = tsne.fit_transform(df[feat_cols].values)
    y    = df["Label"].values

    plt.figure(figsize=(8, 6))
    plt.scatter(X2d[y==1,0], X2d[y==1,1], c="green", alpha=0.5, label="Hey Pakize")
    plt.scatter(X2d[y==0,0], X2d[y==0,1], c="red",   alpha=0.5, label="Negatif")
    plt.legend()
    plt.title("t-SNE: MFCC Feature Space")
    plt.savefig(os.path.join(config.OUTPUT_DIR, "tsne_plot.png"))
    plt.close()
    print("✅ t-SNE grafiği kaydedildi.")