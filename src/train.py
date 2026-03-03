# src/train.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

import os
import shutil


def train():
    # --- 1) Daten laden ---
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # --- 2) Modell trainieren ---
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- 3) Accuracy berechnen ---
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {acc}")

    # --- 4) Zielverzeichnis "models/" vorbereiten ---
    out_dir = "models"

    # Lokalen Ordner löschen falls vorhanden (GitHub Actions braucht sauberen Ordner)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # --- 5) MLflow-Modell speichern ---
    # Wichtig: dieses save_model erzeugt die komplette MLflow-Struktur,
    # die später im CD-Schritt registriert werden kann.
    mlflow.sklearn.save_model(
        sk_model=model,
        path=out_dir
    )

    print(f"MLflow model saved to: {out_dir}")


if __name__ == "__main__":
    train()
