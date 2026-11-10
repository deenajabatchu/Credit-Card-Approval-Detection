
# ============================================================
# CREDIT CARD APPROVAL DETECTION USING MACHINE LEARNING
# Algorithm: Random Forest Classifier
# Dataset: CRX Credit Approval Dataset
# ============================================================

# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 2. Load Dataset
# ============================================================

print("\n========== CREDIT CARD APPROVAL DETECTION ==========\n")

data_path = Path("data") / "crx.csv"

# Original CRX dataset has 16 columns without headers
columns = [f"A{i}" for i in range(1, 17)]

df = pd.read_csv(
    data_path,
    header=None,
    names=columns,
    na_values="?"
)

print("Dataset loaded successfully!")

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

# ============================================================
# 3. Data Preparation
# ============================================================

print("\n========== DATA PREPARATION ==========\n")

# Convert numerical columns into numeric format
numeric_features = ["A2", "A3", "A8", "A11", "A14", "A15"]

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove records without target values
df = df.dropna(subset=["A16"])

# Clean target labels
df["A16"] = df["A16"].astype(str).str.strip()

# Keep only valid target classes
df = df[df["A16"].isin(["+", "-"])].copy()

# Convert target values into numerical labels
# 1 = Approved, 0 = Rejected
df["A16"] = df["A16"].map({
    "+": 1,
    "-": 0
})

print("Missing values in each column:")
print(df.isnull().sum())

print("\nDataset after preparation:")
print(df.head())

print("\nUpdated Dataset Shape:", df.shape)

# ============================================================
# 4. Exploratory Data Analysis (EDA)
# ============================================================

print("\n========== EXPLORATORY DATA ANALYSIS ==========\n")

# Approval and rejection distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    x="A16",
    data=df,
    hue="A16",
    palette="Set2",
    legend=False
)

plt.title("Credit Card Approval Distribution")
plt.xlabel("Application Status (0 = Rejected, 1 = Approved)")
plt.ylabel("Number of Applications")
plt.tight_layout()
plt.show()

# Numerical feature distributions
df[numeric_features].hist(
    figsize=(12, 8),
    bins=20,
    edgecolor="black"
)

plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.show()

# Missing value visualization
plt.figure(figsize=(10, 5))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title("Missing Values in Dataset")
plt.tight_layout()
plt.show()

# ============================================================
# 5. Separate Features and Target
# ============================================================

print("\n========== FEATURE SELECTION ==========\n")

X = df.drop(columns=["A16"])
y = df["A16"]

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

print("Numerical Features:", numeric_features)
print("Categorical Features:", categorical_features)

print("\nTarget Distribution:")
print(y.value_counts())

# ============================================================
# 6. Data Preprocessing
# ============================================================

print("\n========== DATA PREPROCESSING ==========\n")

# Numerical preprocessing
numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])

print("Preprocessing pipeline created successfully!")

# ============================================================
# 7. Train-Test Split
# ============================================================

print("\n========== TRAIN-TEST SPLIT ==========\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])

# ============================================================
# 8. Model Building and Training
# ============================================================

print("\n========== MODEL TRAINING ==========\n")

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

# Train the model
model.fit(X_train, y_train)

print("Random Forest model trained successfully!")

# ============================================================
# 9. Model Prediction
# ============================================================

print("\n========== MODEL PREDICTION ==========\n")

y_pred = model.predict(X_test)

print("Predictions completed successfully!")

# ============================================================
# 10. Model Evaluation
# ============================================================

print("\n========== MODEL EVALUATION ==========\n")

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Rejected", "Approved"],
        zero_division=0
    )
)

# ============================================================
# 11. Confusion Matrix
# ============================================================

print("\n========== CONFUSION MATRIX ==========\n")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")
plt.tight_layout()
plt.show()

# ============================================================
# 12. Feature Importance
# ============================================================

print("\n========== FEATURE IMPORTANCE ==========\n")

# Get transformed feature names
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get Random Forest feature importance
importances = model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(10))

# Visualize top 10 important features
plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False
)

plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ============================================================
# 13. Final Project Summary
# ============================================================

print("\n========== PROJECT SUMMARY ==========\n")

print("Dataset Records:", len(df))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nTarget Label Meaning:")
print("1 = Credit Card Application Approved")
print("0 = Credit Card Application Rejected")

print("\nCredit Card Approval Detection Completed Successfully!")
