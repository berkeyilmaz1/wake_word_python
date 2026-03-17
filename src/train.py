# src/train.py
# Model eğitimi.

import os 
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def run():
    df= pd.read_csv(os.path.join(config.OUTPUT_DIR, "dataset.csv"))
    feat_cols = [c for c in df.columns if c.startswith("F")]
    X = df[feat_cols].values
    y = df["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,stratify=y)

    print(f"Eğitim: {len(X_train)} örnek | Test: {len(X_test)} örnek")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10,100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'linear']
    }

    grid = GridSearchCV(
        SVC(probability=True), param_grid,
        cv=5, scoring="f1", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    print(f"\n✅ En iyi parametreler : {grid.best_params_}")
    print(f"   CV F1               : {grid.best_score_:.4f}\n")
    print(classification_report(y_test, y_pred,
                                 target_names=["Negatif", "Hey Pakize"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negatif", "Hey Pakize"],
                yticklabels=["Negatif", "Hey Pakize"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(best,   os.path.join(config.MODEL_DIR, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(config.MODEL_DIR, "scaler.pkl"))
    print("✅ Model ve scaler kaydedildi.")