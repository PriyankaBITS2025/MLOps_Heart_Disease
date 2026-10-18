
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

PROJECT_DIR = (
    Path.home() / "workspace" / "MLOps_Heart_Disease"
)

DATASET_PATH = (
    PROJECT_DIR / "data" / "raw" / "processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]


def test_dataset_shape():
    df = pd.read_csv(
        DATASET_PATH,
        header=None,
        names=COLUMNS,
        na_values="?"
    )
    assert df.shape == (303, 14)


def test_target_is_binary():
    df = pd.read_csv(
        DATASET_PATH,
        header=None,
        names=COLUMNS,
        na_values="?"
    )
    target = (df["target"] > 0).astype(int)
    assert set(target.unique()) == {0, 1}


def test_saved_models_exist():
    models_dir = PROJECT_DIR / "models"

    assert (models_dir / "logistic_regression.joblib").is_file()
    assert (models_dir / "random_forest.joblib").is_file()


def test_saved_models_predict_binary_classes():
    models_dir = PROJECT_DIR / "models"

    logistic_model = joblib.load(
        models_dir / "logistic_regression.joblib"
    )
    random_forest_model = joblib.load(
        models_dir / "random_forest.joblib"
    )

    df = pd.read_csv(
        DATASET_PATH,
        header=None,
        names=COLUMNS,
        na_values="?"
    )
    X = df.drop(columns=["target"]).iloc[:10]

    for model in [logistic_model, random_forest_model]:
        predictions = model.predict(X)
        assert len(predictions) == 10
        assert set(predictions).issubset({0, 1})
