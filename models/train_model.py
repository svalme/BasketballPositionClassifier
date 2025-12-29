import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pathlib import Path
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "final_dataset.csv"
MODEL_DIR = ROOT / "models"
TEST_SEASON = "2024_25"

DROP_COLS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "NICKNAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "SEASON",
    "LEAGUE",
    "POSITION",
    "POSITION_FINE",
    "POSITION_COARSE"
]

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

train_df = df[df["SEASON"] != TEST_SEASON].copy()
test_df  = df[df["SEASON"] == TEST_SEASON].copy()

x_train = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
x_test  = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])

y_coarse_train = train_df["POSITION_COARSE"]
y_coarse_test  = test_df["POSITION_COARSE"]

y_fine_train = train_df["POSITION_FINE"]
y_fine_test  = test_df["POSITION_FINE"]


# ============================================================
# COARSE MODEL (G / F / C)
# ============================================================

def train_coarse_model(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])
    model.fit(X, y)
    return model


print("\nTraining coarse model...")
coarse_model = train_coarse_model(x_train, y_coarse_train)

coarse_preds = coarse_model.predict(x_test)

print("\n=== COARSE CLASSIFICATION REPORT ===")
print(classification_report(y_coarse_test, coarse_preds))

joblib.dump(coarse_model, f"{MODEL_DIR}/coarse_model.joblib")


# ============================================================
# FINE-GRAINED MODELS
# ============================================================

def train_fine_models(df):
    models = {}

    for group in ["G", "F", "C"]:
        subset = df[df["POSITION_COARSE"] == group]

        X = subset.drop(columns=[c for c in DROP_COLS if c in subset.columns])
        y = subset["POSITION_FINE"]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            ))
        ])

        model.fit(X, y)
        models[group] = model

        print(f"✓ Trained fine model for {group} ({len(subset)} samples)")

    return models


fine_models = train_fine_models(train_df)
joblib.dump(fine_models, f"{MODEL_DIR}/fine_models.joblib")


# ============================================================
# HIERARCHICAL PREDICTION
# ============================================================

def hierarchical_predict(X):
    coarse_preds = coarse_model.predict(X)
    final_preds = []

    for i, coarse in enumerate(coarse_preds):
        model = fine_models[coarse]
        final_preds.append(model.predict(X.iloc[[i]])[0])

    return np.array(final_preds)


# ============================================================
# FINAL EVALUATION
# ============================================================

final_preds = hierarchical_predict(x_test)

print("\n=== 7-CLASS CLASSIFICATION REPORT ===")
print(classification_report(y_fine_test, final_preds))


# Confusion Matrix
labels = sorted(y_fine_test.unique())
cm = confusion_matrix(y_fine_test, final_preds, labels=labels)

disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("7-Class Confusion Matrix")
plt.tight_layout()
plt.show()
