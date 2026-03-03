from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import shutil
import os

def train():
    mlflow.set_experiment("iris-training")

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, random_state=42
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print("Accuracy:", acc)

    # Ordner neu erzeugen
    out_dir = "models"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)

        # speichert MLflow-kompatible Struktur in ./models
        mlflow.sklearn.save_model(sk_model=model, path=out_dir)

if __name__ == "__main__":
    train()
