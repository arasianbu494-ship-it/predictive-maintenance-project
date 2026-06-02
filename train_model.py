import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("data.csv")

print("Dataset Shape:", df.shape)

# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------
df = df.dropna()

# --------------------------------------------------
# IDENTIFY TARGET COLUMN
# --------------------------------------------------
target_col = None

possible_targets = [
    "Machine failure",
    "machine_failure",
    "Failure",
    "failure",
    "Target",
    "target"
]

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError(
        "Target column not found. Please specify your failure column."
    )

print("Target Column:", target_col)

# --------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# --------------------------------------------------
encoders = {}

for col in df.select_dtypes(include="object").columns:

    if col != target_col:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target if needed
if y.dtype == "object":

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    encoders["target"] = target_encoder

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# --------------------------------------------------
# SAVE ENCODERS
# --------------------------------------------------
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# --------------------------------------------------
# SAVE FEATURE NAMES
# --------------------------------------------------
with open("features.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("\nModel Saved Successfully")
print("model.pkl")
print("encoders.pkl")
print("features.pkl")