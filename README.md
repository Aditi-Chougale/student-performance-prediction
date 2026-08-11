# 🎓 Student Performance Prediction

A beginner-friendly Machine Learning project that predicts a student's
performance category — **Poor, Average, Good, or Excellent** — based on
study habits, attendance, and academic history, wrapped in an interactive
**Streamlit** web app.

---

## 📖 Project Description

A student's academic outcome depends on many everyday factors — how much
they study, how regularly they attend class, their past exam scores, and
more. This project trains a **Random Forest Classifier** on student data
to predict which performance category a student is likely to fall into,
and presents the model through a simple, interactive web application.

This project was built as a B.Tech Artificial Intelligence & Machine
Learning mini-project to demonstrate a complete, end-to-end classification
workflow: data preprocessing, model training, evaluation, and deployment.

---

## 🎯 Objective

To build and deploy a classification model that predicts a student's
**Performance Category** based on academic and personal study-related
factors, and to present the model through an easy-to-use web application.

---

## ✨ Features

- Clean, end-to-end ML pipeline (load → clean → validate → train → evaluate → save)
- Random Forest Classification model (4 performance categories)
- Interactive Streamlit web app with number inputs and sliders
- Input validation with friendly error messages
- Instant prediction with a short, encouraging interpretation
- Prediction confidence chart across all categories
- Visualization tabs: performance distribution, feature importance, confusion matrix
- Sidebar with project information
- Well-documented, beginner-friendly code

---

## 🛠️ Technologies Used

| Technology         | Purpose                                  |
|---------------------|--------------------------------------------|
| Python              | Core programming language                |
| Pandas & NumPy      | Data loading and manipulation            |
| Scikit-learn        | Model training and evaluation            |
| Streamlit           | Web application interface                |
| Joblib              | Saving/loading the trained model         |
| Matplotlib/Seaborn  | Visualizations (confusion matrix, feature importance) |

---

## 📊 Dataset Description

The project uses `student_data.csv`, a synthetic but realistic dataset of
**700 student records** generated using a weighted, domain-realistic
scoring formula (study hours, attendance, past scores, sleep, participation,
and backlogs all contribute to the outcome, plus some natural randomness).
It contains the following columns:

| Column               | Description                                          |
|------------------------|--------------------------------------------------------|
| `Study_Hours`          | Average study hours per day                          |
| `Attendance`           | Class attendance percentage                           |
| `Previous_Score`       | Previous exam score (%)                                |
| `Assignment_Score`     | Average assignment score (%)                           |
| `Sleep_Hours`          | Average sleep hours per day                            |
| `Participation`        | Class participation level (1 = Low, 10 = High)         |
| `Backlogs`             | Number of pending backlogs/failed subjects              |
| `Performance`          | **Target** — Poor / Average / Good / Excellent          |

> You can replace `student_data.csv` with your own real-world dataset as
> long as the column names match, or update the code in `train_model.py`
> accordingly.

---

## 🤖 Machine Learning Algorithm

**Random Forest Classifier** (from `scikit-learn`) was chosen because it:
- Works well for multi-class classification problems
- Handles non-linear relationships between features naturally
- Is robust to outliers and doesn't require feature scaling
- Provides feature importance out of the box, useful for interpretation
- Gives strong baseline performance with minimal tuning

---

## 🧹 Data Preprocessing

1. Load the dataset from `student_data.csv`
2. Remove duplicate rows
3. Check and handle missing values (numeric columns filled with the column median)
4. All input features are already numeric, so **no categorical encoding
   is required** for the inputs; the target (`Performance`) is kept as
   readable string labels, which `RandomForestClassifier` supports natively
5. Select the 7 input features and the `Performance` target column
6. Split data into training (80%) and testing (20%) sets, **stratified**
   by class to preserve category balance in both sets

---

## 📈 Model Evaluation

The model is evaluated on a held-out test set using:

- **Accuracy**
- **Precision** (weighted average)
- **Recall** (weighted average)
- **F1 Score** (weighted average)
- **Confusion Matrix**

On the included sample dataset, the model achieves approximately:

```
Accuracy  ≈ 0.71
Precision ≈ 0.71
Recall    ≈ 0.71
F1 Score  ≈ 0.70
```

*(Exact numbers may vary slightly by run/environment. A confusion matrix
image is generated automatically at `confusion_matrix.png` and shown
inside the Streamlit app.)*

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── app.py                        # Streamlit web application
├── train_model.py                # Model training & evaluation script
├── generate_data.py              # Script that created the sample dataset
├── student_data.csv              # Sample dataset
├── student_performance_model.pkl # Trained Random Forest model
├── feature_columns.pkl           # Saved feature column order
├── confusion_matrix.png          # Confusion matrix image (auto-generated)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Files/folders ignored by Git
```

---

## ⚙️ Installation Steps

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/<your-username>/student-performance-prediction.git
   cd student-performance-prediction
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

**Step 1 — Train the model** (this creates `student_performance_model.pkl`,
`feature_columns.pkl`, and `confusion_matrix.png`):
```bash
python train_model.py
```

**Step 2 — Launch the Streamlit web app:**
```bash
streamlit run app.py
```

The app will open automatically in your browser at
`http://localhost:8501`.

---

## 🧪 Example Prediction

**Input:**
| Field | Value |
|---|---|
| Study Hours | 7.0 |
| Attendance | 90% |
| Previous Exam Score | 80% |
| Assignment Score | 85% |
| Sleep Hours | 7.5 |
| Participation | 8 |
| Number of Backlogs | 0 |

**Output:**
```
Predicted Student Performance: Excellent
```
> Outstanding work! This reflects strong study habits, attendance, and
> consistency. Keep it up! 🌟

---

## 🚀 Future Scope

- Train on a larger, real-world student dataset (e.g., UCI Student Performance dataset)
- Add more features such as extracurricular activities, family support, or study environment
- Experiment with other algorithms (Gradient Boosting, XGBoost, SVM) and compare performance
- Add a "what-if" simulator to show how improving one factor changes the predicted category
- Deploy the app publicly using Streamlit Community Cloud
- Add explainability (e.g., SHAP values) for individual predictions

---

## 👩‍💻 Author

**Aditi**
B.Tech Student, Department of Artificial Intelligence & Machine Learning

Team: Pragati Ligade, Padmaja Kothawale, Karuna Bhosale, Saniya Patil
Guide: Mrs. Shamal Desai

---

⭐ If you found this project helpful, consider giving it a star on GitHub!
