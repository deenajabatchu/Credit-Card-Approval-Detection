
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
data_path = Path("data") / "crx.csv"

columns = [f"A{i}" for i in range(1, 17)]
df = pd.read_csv(data_path, header=None, names=columns)

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)

# 2. Data preparation
df = df.replace("?", pd.NA)

# Convert numeric columns
for col in ["A2", "A14"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing target values
df = df.dropna(subset=["A16"])

# Convert approval labels into numbers
df["A16"] = df["A16"].astype(str).str.strip()
df = df[df["A16"].isin(["+", "-"])]
y = df["A16"].map({"+": 1, "-": 0})

X = df.drop(columns=["A16"])

# 3. Identify numeric and categorical columns
numeric_features = ["A2", "A14"]
categorical_features = [
    col for col in X.columns if col not in numeric_features
]

# 4. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Build and train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 7. Model evaluation
predictions = model.predict(X_test)

print("\nModel Evaluation")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nProject completed successfully!")
print("1 = Approved (+)")
print("0 = Rejected (-)")
