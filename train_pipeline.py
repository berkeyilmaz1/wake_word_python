# train_pipeline.py
# BİR KEZ çalıştırılır. Modeli hazırlar.
# Kullanım: python train_pipeline.py

from src import augment, features, train

print("=" * 50)
print("ADIM 1 — Data Augmentation")
print("=" * 50)
augment.run()

print("\n" + "=" * 50)
print("ADIM 2 — Feature Extraction (MFCC → CSV)")
print("=" * 50)
features.run()

print("\n" + "=" * 50)
print("ADIM 3 — SVM Eğitimi")
print("=" * 50)
train.run()

print("\n🎉 Eğitim tamamlandı!")
print("   Arayüzü açmak için: python run_app.py")