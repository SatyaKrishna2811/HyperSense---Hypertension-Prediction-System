"""
Hypertension Prediction System
Complete ML Pipeline: Data Generation → Preprocessing → EDA → Training → Evaluation
"""
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────
def generate_hypertension_dataset(n_samples=2000):
    """Generate a realistic hypertension dataset."""
    n = n_samples
    age = np.random.normal(50, 15, n).clip(18, 90).astype(int)
    bmi = np.random.normal(27, 5, n).clip(16, 50)
    
    # Correlated features
    systolic_bp  = (100 + 0.5*age + 2.5*bmi + np.random.normal(0,10,n)).clip(90,220)
    diastolic_bp = (60  + 0.3*age + 1.5*bmi + np.random.normal(0,8,n)).clip(50,130)
    
    heart_rate       = np.random.normal(72, 12, n).clip(45, 130)
    cholesterol      = np.random.normal(200, 40, n).clip(120, 350)
    blood_glucose    = np.random.normal(100, 25, n).clip(70, 300)
    sodium_intake    = np.random.normal(2300, 600, n).clip(500, 5000)
    physical_activity= np.random.normal(3, 2, n).clip(0, 7)  # days/week
    stress_level     = np.random.randint(1, 11, n)
    sleep_hours      = np.random.normal(7, 1.5, n).clip(3, 10)
    alcohol_units    = np.random.exponential(2, n).clip(0, 20)
    
    smoking_status  = np.random.choice(['Never','Former','Current'], n, p=[0.55,0.25,0.20])
    diabetes        = np.random.choice([0, 1], n, p=[0.85, 0.15])
    family_history  = np.random.choice([0, 1], n, p=[0.60, 0.40])
    kidney_disease  = np.random.choice([0, 1], n, p=[0.92, 0.08])
    on_medication   = np.random.choice([0, 1], n, p=[0.70, 0.30])
    gender          = np.random.choice(['Male','Female'], n)

    # Classify hypertension stage based on BP + risk factors
    def classify_hypertension(row):
        sbp, dbp = row['systolic_bp'], row['diastolic_bp']
        risk = (row['age'] > 60) + (row['bmi'] > 30) + (row['diabetes']) + \
               (row['family_history']) + (row['smoking_status'] == 'Current')
        if sbp < 120 and dbp < 80 and risk < 2:
            return 'Normal'
        elif sbp < 130 and dbp < 80:
            return 'Elevated'
        elif (130 <= sbp < 140) or (80 <= dbp < 90):
            return 'Stage 1 Hypertension'
        elif sbp >= 140 or dbp >= 90:
            if sbp >= 180 or dbp >= 120:
                return 'Hypertensive Crisis'
            return 'Stage 2 Hypertension'
        else:
            return 'Elevated'

    df = pd.DataFrame({
        'age': age, 'gender': gender, 'bmi': bmi.round(1),
        'systolic_bp': systolic_bp.round(0).astype(int),
        'diastolic_bp': diastolic_bp.round(0).astype(int),
        'heart_rate': heart_rate.round(0).astype(int),
        'cholesterol': cholesterol.round(0).astype(int),
        'blood_glucose': blood_glucose.round(0).astype(int),
        'sodium_intake': sodium_intake.round(0).astype(int),
        'physical_activity_days': physical_activity.round(1),
        'stress_level': stress_level,
        'sleep_hours': sleep_hours.round(1),
        'alcohol_units_week': alcohol_units.round(1),
        'smoking_status': smoking_status,
        'diabetes': diabetes,
        'family_history': family_history,
        'kidney_disease': kidney_disease,
        'on_medication': on_medication,
    })

    df['hypertension_stage'] = df.apply(classify_hypertension, axis=1)
    
    # Introduce ~3% missing values
    for col in ['bmi','cholesterol','blood_glucose','sodium_intake','sleep_hours']:
        mask = np.random.random(n) < 0.03
        df.loc[mask, col] = np.nan

    return df

# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_data(df):
    """Clean and prepare data for modelling."""
    print("=== DATA PREPROCESSING ===")
    print(f"Raw shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = ['gender','smoking_status']
    target   = 'hypertension_stage'

    # Impute
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Encode categoricals
    le_gender  = LabelEncoder()
    le_smoking = LabelEncoder()
    df['gender']         = le_gender.fit_transform(df['gender'])
    df['smoking_status'] = le_smoking.fit_transform(df['smoking_status'])

    le_target = LabelEncoder()
    df[target] = le_target.fit_transform(df[target])

    print(f"\nTarget classes: {list(le_target.classes_)}")
    print(f"Class distribution:\n{pd.Series(df[target]).value_counts()}")
    print(f"\nProcessed shape: {df.shape}")
    
    return df, le_target, num_imputer, le_gender, le_smoking

# ─────────────────────────────────────────────
# 3. EDA SUMMARY
# ─────────────────────────────────────────────
def eda_summary(df, le_target):
    """Exploratory Data Analysis — return summary dict."""
    target_map = {i: c for i, c in enumerate(le_target.classes_)}
    stage_counts = df['hypertension_stage'].map(target_map).value_counts().to_dict()

    # Top correlations with target
    corr = df.corr()['hypertension_stage'].abs().sort_values(ascending=False)
    top_features = corr.drop('hypertension_stage').head(8).index.tolist()

    return {
        'total_samples': len(df),
        'features': df.shape[1] - 1,
        'stage_distribution': stage_counts,
        'top_correlated_features': top_features,
        'age_mean': round(df['age'].mean(), 1),
        'bmi_mean': round(df['bmi'].mean(), 1),
        'sbp_mean': round(df['systolic_bp'].mean(), 1),
    }

# ─────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(df, le_target):
    """Train multiple models, return best model + results."""
    target = 'hypertension_stage'
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    print("\n=== MODEL EVALUATION ===")
    for name, model in models.items():
        use_scaled = name == 'Logistic Regression'
        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc  if use_scaled else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cv   = cross_val_score(model, Xtr, y_train, cv=5, scoring='accuracy').mean()

        results[name] = {
            'accuracy': round(acc*100,2),
            'precision': round(prec*100,2),
            'recall': round(rec*100,2),
            'f1_score': round(f1*100,2),
            'cv_accuracy': round(cv*100,2),
        }
        print(f"{name:25s} | Acc: {acc:.3f} | F1: {f1:.3f} | CV: {cv:.3f}")

    # Best model = highest F1
    best_name = max(results, key=lambda k: results[k]['f1_score'])
    best_model = models[best_name]
    print(f"\n✅ Best Model: {best_name}")

    # Feature importance
    feat_imp = {}
    if hasattr(best_model, 'feature_importances_'):
        fi = best_model.feature_importances_
        feat_imp = dict(zip(X.columns, [round(float(v)*100,2) for v in fi]))
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x:-x[1])[:10])

    # Confusion matrix
    use_scaled = best_name == 'Logistic Regression'
    Xte = X_test_sc if use_scaled else X_test
    y_pred_best = best_model.predict(Xte)
    cm = confusion_matrix(y_test, y_pred_best).tolist()
    class_names = list(le_target.classes_)

    return best_model, best_name, scaler, X.columns.tolist(), results, feat_imp, cm, class_names, X_train, y_train

# ─────────────────────────────────────────────
# 5. SAVE ARTEFACTS
# ─────────────────────────────────────────────
def save_artifacts(model, scaler, le_target, le_gender, le_smoking, num_imputer, feature_names, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model,       f"{models_dir}/best_model.pkl")
    joblib.dump(scaler,      f"{models_dir}/scaler.pkl")
    joblib.dump(le_target,   f"{models_dir}/le_target.pkl")
    joblib.dump(le_gender,   f"{models_dir}/le_gender.pkl")
    joblib.dump(le_smoking,  f"{models_dir}/le_smoking.pkl")
    joblib.dump(num_imputer, f"{models_dir}/num_imputer.pkl")
    with open(f"{models_dir}/feature_names.json","w") as f:
        json.dump(feature_names, f)
    print(f"\n💾 Artifacts saved to {models_dir}")

# ─────────────────────────────────────────────
# 6. PREDICTION PIPELINE
# ─────────────────────────────────────────────
def predict_hypertension(patient_data: dict, models_dir: str):
    """Load model and predict for a new patient."""
    model      = joblib.load(f"{models_dir}/best_model.pkl")
    le_target  = joblib.load(f"{models_dir}/le_target.pkl")
    le_gender  = joblib.load(f"{models_dir}/le_gender.pkl")
    le_smoking = joblib.load(f"{models_dir}/le_smoking.pkl")

    with open(f"{models_dir}/feature_names.json") as f:
        feature_names = json.load(f)

    row = patient_data.copy()
    row['gender']         = le_gender.transform([row['gender']])[0]
    row['smoking_status'] = le_smoking.transform([row['smoking_status']])[0]

    X = pd.DataFrame([row])[feature_names]
    pred_idx  = model.predict(X)[0]
    pred_prob = model.predict_proba(X)[0]
    stage     = le_target.inverse_transform([pred_idx])[0]
    confidence = round(max(pred_prob)*100, 1)
    class_probs = {le_target.inverse_transform([i])[0]: round(float(p)*100,1)
                   for i, p in enumerate(pred_prob)}

    return stage, confidence, class_probs

# ─────────────────────────────────────────────
# 7. RECOMMENDATION MODULE
# ─────────────────────────────────────────────
def get_recommendations(stage, patient_data):
    base = [
        "Monitor blood pressure regularly and maintain a health diary.",
        "Limit sodium intake to under 2,300 mg/day (ideally 1,500 mg).",
        "Aim for at least 150 minutes of moderate aerobic activity per week.",
        "Maintain a balanced diet rich in fruits, vegetables, and whole grains (DASH diet).",
        "Manage stress through mindfulness, yoga, or relaxation techniques.",
        "Ensure 7–9 hours of quality sleep per night.",
    ]
    stage_recs = {
        'Normal': [
            "Great news! Your blood pressure is in the healthy range.",
            "Continue your current healthy lifestyle habits.",
            "Schedule an annual check-up to maintain your health.",
        ],
        'Elevated': [
            "Your blood pressure is slightly above optimal — take preventive action now.",
            "Reduce alcohol consumption and quit smoking if applicable.",
            "Consider the DASH (Dietary Approaches to Stop Hypertension) diet.",
        ],
        'Stage 1 Hypertension': [
            "Consult your doctor about lifestyle changes and possible medication.",
            "Monitor blood pressure daily and log readings.",
            "Reduce caffeine intake and avoid NSAIDs without medical advice.",
        ],
        'Stage 2 Hypertension': [
            "⚠️ Seek medical attention promptly for treatment options.",
            "Medication is likely required — do not delay consultation.",
            "Strictly limit sodium, saturated fats, and alcohol.",
            "Monitor for symptoms: severe headache, chest pain, shortness of breath.",
        ],
        'Hypertensive Crisis': [
            "🚨 SEEK EMERGENCY MEDICAL CARE IMMEDIATELY.",
            "Do not drive yourself — call emergency services.",
            "A hypertensive crisis (BP ≥180/120) can cause organ damage.",
            "After stabilization, long-term medication management is essential.",
        ],
    }
    personal = []
    if patient_data.get('bmi',0) > 30:
        personal.append("Weight management: losing even 5–10% of body weight can lower BP significantly.")
    if patient_data.get('smoking_status','Never') == 'Current':
        personal.append("Smoking cessation is one of the most impactful steps you can take for cardiovascular health.")
    if patient_data.get('stress_level',0) > 7:
        personal.append("Your stress level is high. Consider speaking with a mental health professional.")
    if patient_data.get('physical_activity_days',7) < 3:
        personal.append("Increase physical activity — even 30 minutes of walking daily makes a difference.")
    if patient_data.get('sodium_intake',0) > 3000:
        personal.append("Your sodium intake is high. Avoid processed foods and check nutrition labels.")

    return {
        'stage_specific': stage_recs.get(stage, []),
        'general': base,
        'personalized': personal,
    }

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # ── Cross-platform paths: creates folders next to this script ──
    from pathlib import Path
    BASE_DIR   = Path(__file__).resolve().parent
    DATA_DIR   = BASE_DIR / 'data'
    MODELS_DIR = str(BASE_DIR / 'models')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / 'models').mkdir(parents=True, exist_ok=True)

    print("🔬 Hypertension Prediction System — Training Pipeline")
    print("="*55)
    print(f"📁 Project folder: {BASE_DIR}")

    # Generate data
    print("\n[1/5] Generating dataset...")
    df_raw = generate_hypertension_dataset(2000)
    df_raw.to_csv(DATA_DIR / 'hypertension_dataset.csv', index=False)
    print(f"Dataset saved: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    # Preprocess
    print("\n[2/5] Preprocessing...")
    df_proc, le_target, num_imputer, le_gender, le_smoking = preprocess_data(df_raw.copy())

    # EDA
    print("\n[3/5] EDA Summary...")
    eda = eda_summary(df_proc, le_target)
    with open(DATA_DIR / 'eda_summary.json', 'w') as f:
        json.dump(eda, f, indent=2)
    print(json.dumps(eda, indent=2))

    # Train
    print("\n[4/5] Training models...")
    best_model, best_name, scaler, feature_names, results, feat_imp, cm, class_names, X_train, y_train = \
        train_and_evaluate(df_proc, le_target)

    # Save artefacts
    print("\n[5/5] Saving artifacts...")
    save_artifacts(best_model, scaler, le_target, le_gender, le_smoking, num_imputer, feature_names, MODELS_DIR)

    # Save results for UI
    pipeline_results = {
        'best_model': best_name,
        'model_results': results,
        'feature_importance': feat_imp,
        'confusion_matrix': cm,
        'class_names': class_names,
        'eda': eda,
        'feature_names': feature_names,
    }
    with open(DATA_DIR / 'pipeline_results.json', 'w') as f:
        json.dump(pipeline_results, f, indent=2)

    # Quick prediction test
    print("\n=== TEST PREDICTION ===")
    test_patient = {
        'age': 58, 'gender': 'Male', 'bmi': 32.5,
        'systolic_bp': 155, 'diastolic_bp': 98,
        'heart_rate': 82, 'cholesterol': 240, 'blood_glucose': 130,
        'sodium_intake': 3200, 'physical_activity_days': 1,
        'stress_level': 8, 'sleep_hours': 5.5, 'alcohol_units_week': 10,
        'smoking_status': 'Current', 'diabetes': 1,
        'family_history': 1, 'kidney_disease': 0, 'on_medication': 0,
    }
    stage, conf, probs = predict_hypertension(test_patient, MODELS_DIR)
    recs  = get_recommendations(stage, test_patient)
    print(f"Stage: {stage} | Confidence: {conf}%")
    print(f"Probabilities: {probs}")
    print(f"Recommendations: {recs['stage_specific'][:2]}")
    print("\n✅ Pipeline complete!")
