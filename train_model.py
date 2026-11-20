import pandas as pd

# Load customer churn dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Display first five records
print("\nCUSTOMER DATASET")
print(df.head())

# Dataset information
print("\nDATASET SHAPE")
print(df.shape)

# Column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Dataset information
print("\nDATASET INFORMATION")
print(df.info())

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Check churn distribution
print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts())