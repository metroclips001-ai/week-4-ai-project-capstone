# Week 4 Capstone: End-to-End Machine Learning Project
# Train a classification model, preprocess data, evaluate it,
# and serialize the trained pipeline using Joblib.

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "breast_cancer_model.joblib"


def train_and_save_model():
    """Load data, preprocess it, train the model, evaluate it, and save it."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Keep preprocessing and the model together to prevent training/serving mismatch.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=data.target_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions)

    print("=" * 65)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 65)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    print(f"Accuracy        : {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

    # Save the complete preprocessing + model pipeline.
    joblib.dump(pipeline, MODEL_PATH)

    pd.DataFrame(matrix).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv", index=False, header=False
    )

    pd.DataFrame([{
        "Model": "Logistic Regression",
        "Accuracy": accuracy,
        "Training Samples": len(X_train),
        "Testing Samples": len(X_test),
    }]).to_csv(
        OUTPUT_DIR / "model_metrics.csv", index=False
    )

    print(f"\nSaved model to: {MODEL_PATH.resolve()}")
    return pipeline, data


if __name__ == "__main__":
    train_and_save_model()
