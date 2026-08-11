"""
generate_data.py
-----------------
Creates a realistic, synthetic 'student_data.csv' dataset for the
Student Performance Prediction project. Run this only if you want to
regenerate the dataset — a ready-made student_data.csv is already
included in the project.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 700
rows = []

for _ in range(N):
    study_hours = round(np.clip(np.random.normal(5, 2.2), 0, 12), 1)
    attendance = round(np.clip(np.random.normal(75, 15), 30, 100), 1)
    previous_score = round(np.clip(np.random.normal(65, 15), 20, 100), 1)
    assignment_score = round(np.clip(np.random.normal(70, 14), 20, 100), 1)
    sleep_hours = round(np.clip(np.random.normal(6.5, 1.3), 3, 10), 1)
    participation = int(np.clip(np.random.normal(6, 2), 1, 10))
    backlogs = int(np.random.choice([0, 1, 2, 3, 4], p=[0.55, 0.25, 0.12, 0.05, 0.03]))

    # ---- Composite performance score (weighted, realistic) ----
    # Sleep has a sweet spot around 7-8 hours; too little or too much hurts.
    sleep_penalty = abs(sleep_hours - 7.5) * 2.5

    score = (
        study_hours * 4.5
        + attendance * 0.35
        + previous_score * 0.30
        + assignment_score * 0.25
        + participation * 2.0
        - backlogs * 8.0
        - sleep_penalty
    )

    # small random noise to avoid an overly clean (unrealistic) boundary
    score += np.random.normal(0, 2)

    # ---- Bucket the composite score into performance categories ----
    # Thresholds chosen (based on the score distribution) to give a
    # reasonably balanced spread across all four categories.
    if score < 72:
        performance = "Poor"
    elif score < 88:
        performance = "Average"
    elif score < 104:
        performance = "Good"
    else:
        performance = "Excellent"

    rows.append({
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Score": previous_score,
        "Assignment_Score": assignment_score,
        "Sleep_Hours": sleep_hours,
        "Participation": participation,
        "Backlogs": backlogs,
        "Performance": performance,
    })

df = pd.DataFrame(rows)

# Introduce a few missing values on purpose so the training script
# demonstrates real missing-value handling (kept small & realistic)
missing_idx = np.random.choice(df.index, size=10, replace=False)
df.loc[missing_idx, "Attendance"] = np.nan

df.to_csv("student_data.csv", index=False)
print(f"student_data.csv created with {len(df)} rows.")
print(df["Performance"].value_counts())
print(df.head())
