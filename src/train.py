# src/train.py
# Trainiert ein simples Iris-Modell, loggt es als MLflow-Modell
# und materialisiert das geloggte Artefakt nach ./models (für CD).

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

from mlflow.artifacts import download_artifacts
import os
import shutil


def main():
    # --- 1) Daten laden ---
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 2) Modell trainieren ---
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # --- 3) Accuracy ---
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[train.py] Accuracy: {acc:.4f}")

    # --- 4) MLflow-Run + Modell LOGGEN ---
    # Hinweis: 'artifact_path' ist in neueren MLflow-Versionen deprecated -> 'name' verwenden.
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )
        mlflow.log_metric("acc", acc)
        run_id = run.info.run_id
        print(f"[train.py] MLflow run_id: {run_id}")

        # Das geloggte Artefakt (Ordner 'model') aus dem Run sauber lokal herunterladen
        local_model_dir = download_artifacts(run_id=run_id, artifact_path="model")
        print(f"[train.py] Downloaded run artifact to: {local_model_dir}")

    # --- 5) Artefakt an den CI/CD-Ort ./models materialisieren ---
    dst_dir = "models"

    # Sauberer Zustand
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    shutil.copytree(local_model_dir, dst_dir)
    print(f"[train.py] Materialized MLflow model to: {dst_dir}")

    # --- 6) Minimalprüfung: MLmodel & conda.yaml vorhanden? ---
    mlmodel_path = os.path.join(dst_dir, "MLmodel")
    conda_path = os.path.join(dst_dir, "conda.yaml")

    missing = []
    if not os.path.exists(mlmodel_path):
        missing.append("MLmodel")
    if not os.path.exists(conda_path):
        missing.append("conda.yaml")

    if missing:
        print(f"[train.py][WARN] Missing expected files in '{dst_dir}': {missing}")
    else:
        print(f"[train.py] OK: '{dst_dir}' enthält MLmodel & conda.yaml")


if __name__ == "__main__":
    main()
