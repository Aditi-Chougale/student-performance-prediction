"""
app.py
-------
Streamlit web application for the Student Performance Prediction project.
Loads the trained Random Forest Classifier (saved by train_model.py) and
predicts a student's performance category based on user-provided inputs.

Run:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

MODEL_PATH = "student_performance_model.pkl"
COLUMNS_PATH = "feature_columns.pkl"
DATA_PATH = "student_data.csv"
CONFUSION_MATRIX_IMG = "confusion_matrix.png"

CLASS_ORDER = ["Poor", "Average", "Good", "Excellent"]

# Short, positive/encouraging interpretation shown alongside the prediction
INTERPRETATIONS = {
    "Poor": (
        "There's a lot of room to grow. Increasing study hours, improving "
        "attendance, and clearing backlogs can make a big difference. 💪"
    ),
    "Average": (
        "A steady foundation! With a bit more consistency in study time and "
        "participation, this can move up to Good or Excellent. 📈"
    ),
    "Good": (
        "Solid performance! Keep up the consistent effort — a few more study "
        "hours and higher participation could push this to Excellent. 👏"
    ),
    "Excellent": (
        "Outstanding work! This reflects strong study habits, attendance, "
        "and consistency. Keep it up! 🌟"
    ),
}


@st.cache_resource
def load_artifacts():
    """Load the trained model and feature-column order."""
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    return model, feature_columns


@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def main():
    st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="centered")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("📋 Project Info")
        st.markdown("**Project Name:** Student Performance Prediction")
        st.markdown(
            "**Description:** A Machine Learning web app that predicts a "
            "student's performance category (Poor / Average / Good / "
            "Excellent) based on study habits, attendance, and academic "
            "history."
        )
        st.markdown("**Technologies Used:**")
        st.markdown(
            "- Python\n"
            "- Pandas & NumPy\n"
            "- Scikit-learn\n"
            "- Streamlit\n"
            "- Joblib\n"
            "- Matplotlib / Seaborn"
        )
        st.markdown("**ML Algorithm:** Random Forest Classifier")
        st.markdown("---")
        st.caption("B.Tech AI & ML Mini Project")

    # ---------------- Main title ----------------
    st.title("🎓 Student Performance Prediction")
    st.write("Enter the student's details below to predict their performance category.")

    # ---------------- Load model artifacts ----------------
    try:
        model, feature_columns = load_artifacts()
    except FileNotFoundError:
        st.error(
            "⚠️ Trained model files were not found.\n\n"
            "Please run `python train_model.py` first to train and save the model, "
            "then restart this app."
        )
        st.stop()

    st.subheader("Student Details")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "Study Hours (per day)", min_value=0.0, max_value=16.0, value=5.0, step=0.5
        )
        attendance = st.number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0
        )
        previous_score = st.number_input(
            "Previous Exam Score (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0
        )
        assignment_score = st.number_input(
            "Assignment Score (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0
        )

    with col2:
        sleep_hours = st.number_input(
            "Sleep Hours (per day)", min_value=0.0, max_value=14.0, value=7.0, step=0.5
        )
        participation = st.slider(
            "Class Participation (1 = Low, 10 = High)", min_value=1, max_value=10, value=6
        )
        backlogs = st.number_input(
            "Number of Backlogs", min_value=0, max_value=20, value=0, step=1
        )

    st.write("")
    predict_clicked = st.button("Predict Performance", type="primary", use_container_width=True)

    if predict_clicked:
        # ---------------- Input validation ----------------
        errors = []

        if attendance < 0 or attendance > 100:
            errors.append("Attendance must be between 0 and 100%.")
        if previous_score < 0 or previous_score > 100:
            errors.append("Previous Exam Score must be between 0 and 100%.")
        if assignment_score < 0 or assignment_score > 100:
            errors.append("Assignment Score must be between 0 and 100%.")
        if study_hours < 0:
            errors.append("Study Hours cannot be negative.")
        if sleep_hours < 0:
            errors.append("Sleep Hours cannot be negative.")
        if sleep_hours > 16:
            errors.append("Sleep Hours seems unrealistically high (max 16).")
        if backlogs < 0:
            errors.append("Number of Backlogs cannot be negative.")

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            st.stop()

        # ---------------- Build input row (same feature order as training) ----------------
        input_dict = {
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Score": previous_score,
            "Assignment_Score": assignment_score,
            "Sleep_Hours": sleep_hours,
            "Participation": participation,
            "Backlogs": backlogs,
        }
        input_df = pd.DataFrame([input_dict])[feature_columns]

        try:
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            prob_df = pd.DataFrame(
                {"Category": model.classes_, "Confidence": probabilities}
            ).sort_values("Confidence", ascending=False)

            st.success("✅ Prediction complete!")
            st.markdown(f"### 🎯 Predicted Student Performance: **{prediction}**")
            st.info(INTERPRETATIONS.get(prediction, ""))

            st.write("**Prediction Confidence by Category:**")
            st.bar_chart(prob_df.set_index("Category"))
        except Exception as e:
            st.error(f"⚠️ Something went wrong while predicting: {e}")

    # ---------------- Visualization Section ----------------
    st.markdown("---")
    st.subheader("📊 Dataset & Model Insights")

    df = load_dataset()

    viz_tab1, viz_tab2, viz_tab3 = st.tabs(
        ["Performance Distribution", "Feature Importance", "Confusion Matrix"]
    )

    with viz_tab1:
        if df is not None and "Performance" in df.columns:
            counts = df["Performance"].value_counts().reindex(CLASS_ORDER).fillna(0)
            st.bar_chart(counts)
            st.caption("Distribution of performance categories in the training dataset.")
        else:
            st.info("Dataset not found — run `python train_model.py` after placing 'student_data.csv' in the project folder.")

    with viz_tab2:
        try:
            importances = model.feature_importances_
            fi_df = pd.DataFrame(
                {"Feature": feature_columns, "Importance": importances}
            ).sort_values("Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(fi_df["Feature"], fi_df["Importance"], color="#4C72B0")
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance (Random Forest)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.info(f"Feature importance not available: {e}")

    with viz_tab3:
        if os.path.exists(CONFUSION_MATRIX_IMG):
            st.image(CONFUSION_MATRIX_IMG, caption="Confusion Matrix on test data (from train_model.py)")
        else:
            st.info("Confusion matrix image not found — run `python train_model.py` to generate it.")


if __name__ == "__main__":
    main()
