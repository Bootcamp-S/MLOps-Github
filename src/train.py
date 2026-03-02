from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def train():
    mlflow.set_experiment("iris-training")

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train()
