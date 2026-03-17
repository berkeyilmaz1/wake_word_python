# src/features.py
# Tüm ses dosyalarından MFCC çıkarır → dataset.csv oluşturur.
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