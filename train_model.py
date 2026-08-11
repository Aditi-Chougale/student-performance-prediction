"""
train_model.py
---------------
Trains a Random Forest Classifier to predict a student's performance
category (Poor / Average / Good / Excellent).

Steps performed:
    1. Load the dataset
    2. Check and handle missing values
    3. Encode categorical features (target only, all inputs are numeric)
    4. Select input features and target
    5. Train-test split
    6. Train the model
    7. Evaluate: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
    8. Save the trained model with Joblib

Run:
    python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "student_data.csv"
MODEL_PATH = "student_performance_model.pkl"
COLUMNS_PATH = "feature_columns.pkl"
CONFUSION_MATRIX_IMG = "confusion_matrix.png"

FEATURE_COLUMNS = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignment_Score",
    "Sleep_Hours",
    "Participation",
    "Backlogs",
]
TARGET_COLUMN = "Performance"
CLASS_ORDER = ["Poor", "Average", "Good", "Excellent"]


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading dataset from '{path}' ...")
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data ...")
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # Drop rows where the target itself is missing
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("Handling missing values ...")
    missing_before = df[FEATURE_COLUMNS].isnull().sum().sum()
    for col in FEATURE_COLUMNS:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled missing values in '{col}' with median = {median_val}")
    print(f"Total missing values fixed: {missing_before}")
    return df


def main():
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Clean + handle missing values
    df = clean_data(df)
    df = handle_missing_values(df)

    # 3 & 4. Features / target (no encoding needed — all input features
    # are already numeric; the target stays as readable string labels,
    # which scikit-learn's RandomForestClassifier handles natively)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 5. Train-test split (stratified to keep class balance in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 6. Train the model
    print("Training RandomForestClassifier ...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    print("\n----- Model Evaluation -----")
    print(f"Accuracy  : {accuracy:.3f}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1:.3f}")
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("-----------------------------\n")

    # Save a confusion matrix heatmap image (used by the Streamlit app)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMG)
    plt.close()
    print(f"Confusion matrix image saved to: {CONFUSION_MATRIX_IMG}")

    # 8. Save model + feature column order
    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, COLUMNS_PATH)

    print(f"Model saved to      : {MODEL_PATH}")
    print(f"Feature order saved : {COLUMNS_PATH}")
    print("\nTraining complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
