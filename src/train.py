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
    print(f"Model accuracy: {acc:.4f}")

    # --- 4) MLflow-Run starten und Modell LOGGEN ---
    # Hinweis: Ab MLflow 3.x ist 'artifact_path' deprecated. Verwende 'name'.
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"  # erzeugt artifacts unter .../artifacts/model/
        )
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")

        # --- 5) Artefakt sauber nach ./models materialisieren ---
        # Nutze die 'run_id' + 'artifact_path', NICHT den rohen artifact_uri-String.
        local_model_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model"
        )

    dst_path = "models"
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)

    shutil.copytree(local_model_dir, dst_path)

    # Optional: Mini-Check
    expected_files = ["MLmodel", "conda.yaml"]
    missing = [f for f in expected_files if not os.path.exists(os.path.join(dst_path, f))]
    if missing:
        print(f"Warnung: Folgende erwartete Dateien fehlen in '{dst_path}': {missing}")
    else:
        print(f"MLflow model materialized to: {dst_path} (inkl. MLmodel & conda.yaml)")


if __name__ == "__main__":
    train()
``
