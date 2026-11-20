# ==========================================
# CUSTOMER CHURN PREDICTION
# RANDOM FOREST MACHINE LEARNING MODEL
# ==========================================

import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("CUSTOMER CHURN MODEL TRAINING STARTED")

# STEP 1: LOAD PREPARED DATA

X_train, X_test, y_train, y_test = joblib.load(
    "models/prepared_data.pkl"
)

print("\nDATA LOADED SUCCESSFULLY")

# STEP 2: CREATE RANDOM FOREST MODEL

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# STEP 3: TRAIN MODEL

print("\nTRAINING RANDOM FOREST MODEL...")

model.fit(X_train, y_train)

print("MODEL TRAINING COMPLETED")

# STEP 4: MAKE PREDICTIONS

y_pred = model.predict(X_test)

# STEP 5: CALCULATE PERFORMANCE

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nMODEL PERFORMANCE")
print("--------------------------")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

# STEP 6: CLASSIFICATION REPORT

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

# STEP 7: CONFUSION MATRIX

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# STEP 8: SAVE TRAINED MODEL

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/churn_model.pkl")

# STEP 9: SAVE MODEL METRICS

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

joblib.dump(metrics, "models/model_metrics.pkl")

# STEP 10: FEATURE IMPORTANCE

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

importance = model.feature_importances_

importance_data = sorted(
    zip(feature_columns, importance),
    key=lambda x: x[1],
    reverse=True
)

print("\nTOP 10 CHURN FACTORS")

for feature, score in importance_data[:10]:
    print(f"{feature}: {score:.4f}")

print("\n================================")
print("RANDOM FOREST MODEL COMPLETED")
print("================================")
print("MODEL SAVED SUCCESSFULLY!")