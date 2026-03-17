import os

# ── Klasör yolları ─────────────────────────────────────────
RAW_POS_DIR  = "data/raw/positive"
RAW_NEG_DIR  = "data/raw/negative"
AUG_POS_DIR  = "data/augmented/positive"
AUG_NEG_DIR  = "data/augmented/negative"
OUTPUT_DIR   = "output"
MODEL_DIR    = "output/models"

# ── Ses ayarları ───────────────────────────────────────────
SAMPLE_RATE      = 22050   # Hz 
TARGET_DURATION  = 1.5     # saniye padding ve cropping için
N_MFCC           = 20      # MFCC katsayısı

# ── Data augmentation ayarları ─────────────────────────────
AUGMENT_COPIES   = 3       # her orijinal kayıttan kaç varyasyon üretilsin
NOISE_LEVEL      = 0.02    # white noise şiddeti (%3)
PITCH_RANGE      = 1       # yarım ton (±2)
SPEED_RANGE      = (0.9, 1.1)  # hız aralığı

# ── Model ayarları ─────────────────────────────────────────
TEST_SIZE        = 0.2     # verinin %20'si test için ayrılır
RANDOM_STATE     = 42      # tekrarlanabilir sonuçlar için sabit seed