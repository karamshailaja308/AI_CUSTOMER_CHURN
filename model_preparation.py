# ==========================================
# CUSTOMER CHURN PREDICTION
# DATA PREPROCESSING AND MODEL PREPARATION
# ==========================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("CUSTOMER CHURN PROJECT STARTED")

# STEP 1: LOAD DATASET

df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("\nDATASET LOADED SUCCESSFULLY")
print("Original Dataset Shape:", df.shape)

# STEP 2: DISPLAY DATASET

print("\nFIRST FIVE CUSTOMER RECORDS")
print(df.head())

# STEP 3: REMOVE UNNECESSARY COLUMN

df = df.drop("customerID", axis=1)

# STEP 4: CONVERT TOTAL CHARGES TO NUMERIC

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# STEP 5: HANDLE MISSING VALUES

print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())

df = df.dropna().copy()

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum().sum())

# STEP 6: ENCODE TARGET COLUMN

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# STEP 7: SEPARATE FEATURES AND TARGET

X = df.drop("Churn", axis=1)
y = df["Churn"]

# STEP 8: ENCODE CATEGORICAL FEATURES

X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

print("\nFEATURE ENCODING COMPLETED")
print("Total Features:", X.shape[1])

# STEP 9: SPLIT TRAINING AND TESTING DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDATASET SPLIT COMPLETED")
print("Training Records:", X_train.shape)
print("Testing Records:", X_test.shape)

# STEP 10: FEATURE SCALING

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 11: CREATE MODEL FOLDER

os.makedirs("models", exist_ok=True)

# STEP 12: SAVE SCALER AND FEATURE NAMES

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

# STEP 13: SAVE PREPARED DATA

joblib.dump(
    (X_train_scaled, X_test_scaled, y_train, y_test),
    "models/prepared_data.pkl"
)

# STEP 14: FINAL OUTPUT

print("\n================================")
print("DATA PREPARATION COMPLETED")
print("================================")

print("Training Data:", X_train_scaled.shape)
print("Testing Data:", X_test_scaled.shape)

print("\nCHURN DISTRIBUTION")
print(y.value_counts())

print("\nALL DATA IS READY FOR MACHINE LEARNING!")