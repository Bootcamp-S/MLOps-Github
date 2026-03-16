# train.py (Ausschnitt)
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LogisticRegression(max_iter=200).fit(X_train, y_train)

with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",     # Unterordner im Run
        registered_model_name=None # optional
    )
    mlflow.log_metric("acc", model.score(X_test, y_test))

# Optional: Artefakt aus dem Run exportieren (CI-Schritt), z. B. nach ./models/
``
