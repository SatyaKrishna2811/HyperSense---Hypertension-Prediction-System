# HyperSense — Hypertension Prediction System
### Machine Learning Project Documentation

---

## Project Overview
HyperSense is an advanced ML system that predicts and classifies hypertension stages
(Normal, Elevated, Stage 1/2, Hypertensive Crisis) from 18 clinical and lifestyle parameters.

**This is a decision-support tool only. Not a substitute for medical diagnosis.**

---

## Project Structure
```
hypertension_project/
├── generate_and_train.py   # Full ML pipeline (data → model)
├── app.html                # Web UI (standalone, open in browser)
├── data/
│   ├── hypertension_dataset.csv   # Synthetic training data (2000 rows)
│   ├── eda_summary.json           # EDA results
│   └── pipeline_results.json      # Model evaluation results
├── models/
│   ├── best_model.pkl        # Trained Decision Tree
│   ├── scaler.pkl            # StandardScaler
│   ├── le_target.pkl         # Label encoder (target)
│   ├── le_gender.pkl         # Label encoder (gender)
│   ├── le_smoking.pkl        # Label encoder (smoking)
│   ├── num_imputer.pkl       # Median imputer
│   └── feature_names.json    # Feature order
└── README.md
```

---

## Features (18 Clinical Parameters)

| Category       | Feature                  | Type       |
|----------------|--------------------------|------------|
| Demographics   | Age, Gender              | Numeric/Cat|
| Vitals         | BMI, Heart Rate          | Numeric    |
| Blood Pressure | Systolic BP, Diastolic BP| Numeric    |
| Labs           | Cholesterol, Blood Glucose, Sodium Intake | Numeric |
| Lifestyle      | Physical Activity, Stress Level, Sleep Hours, Alcohol Units, Smoking Status | Mixed |
| Medical History| Diabetes, Family History, Kidney Disease, On Medication | Binary |

---

## Hypertension Classification (AHA 2017)

| Stage                 | Systolic (mmHg) | Diastolic (mmHg) |
|-----------------------|-----------------|------------------|
| Normal                | < 120           | < 80             |
| Elevated              | 120–129         | < 80             |
| Stage 1 Hypertension  | 130–139         | 80–89            |
| Stage 2 Hypertension  | ≥ 140           | ≥ 90             |
| Hypertensive Crisis   | ≥ 180           | ≥ 120            |

---

## ML Pipeline

### Step 1 — Data Generation
- 2,000 synthetic patient records with realistic correlations
- BP values correlated with age, BMI, diabetes, family history
- 3% random missing values introduced for realism

### Step 2 — Preprocessing
- Median imputation for missing numerical values
- Label encoding for categorical features (gender, smoking status)
- StandardScaler for Logistic Regression

### Step 3 — Models Trained

| Model               | Accuracy | F1-Score | CV Accuracy |
|---------------------|----------|----------|-------------|
| Logistic Regression | 95.5%    | 95.4%    | 95.2%       |
| Decision Tree       | 100.0%   | 100.0%   | 99.6%       |
| Random Forest       | 99.3%    | 99.1%    | 99.1%       |
| Gradient Boosting   | 100.0%   | 100.0%   | 99.9%       |

**Best Model: Decision Tree** (highest F1, interpretable)

### Step 4 — Prediction Pipeline
```python
from generate_and_train import predict_hypertension, get_recommendations

patient = {
    'age': 55, 'gender': 'Female', 'bmi': 28.0,
    'systolic_bp': 138, 'diastolic_bp': 88,
    'heart_rate': 76, 'cholesterol': 215, 'blood_glucose': 105,
    'sodium_intake': 2400, 'physical_activity_days': 3,
    'stress_level': 5, 'sleep_hours': 7.0, 'alcohol_units_week': 3,
    'smoking_status': 'Never', 'diabetes': 0,
    'family_history': 1, 'kidney_disease': 0, 'on_medication': 0,
}

stage, confidence, probs = predict_hypertension(patient, './models')
recs = get_recommendations(stage, patient)
print(f"Stage: {stage} | Confidence: {confidence}%")
```

---

## How to Run

### Requirements
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Train the Model
```bash
python generate_and_train.py
```

### Open the Web App
Simply open `app.html` in any modern browser — **no server required**.

---

## Use Cases
- **Preventive Screening**: Early detection for at-risk populations
- **Hypertensive Monitoring**: Ongoing tracking for known hypertension patients
- **Emergency Triage**: Quick classification in clinical settings

---

## Ethical Use Disclaimer
This system is for **research and educational purposes only**. 
Predictions are based on a statistical model and should never replace clinical judgement.
Always consult a licensed healthcare provider for medical decisions.

---

*Built with Python 3.12 · scikit-learn · pandas · numpy*
