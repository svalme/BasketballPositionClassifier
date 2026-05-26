import json
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
ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / "data"    / "final_dataset.csv"
MODEL_DIR   = ROOT / "models"
RESULTS_DIR = ROOT / "results"
TEST_SEASON = "2024_25"

RESULTS_DIR.mkdir(exist_ok=True)

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

# Save feature columns so inference.py can load the exact list used at training time
feature_cols = x_train.columns.tolist()
with open(MODEL_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"Feature columns saved: {len(feature_cols)} features")

y_coarse_train = train_df["POSITION_COARSE"]
y_coarse_test  = test_df["POSITION_COARSE"]

y_fine_train = train_df["POSITION_FINE"]
y_fine_test  = test_df["POSITION_FINE"]


# ============================================================
# COARSE MODEL (G / F / C)
# ============================================================

def train_coarse_model(x, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        ))
    ])
    model.fit(x, y)
    return model


print("\nTraining coarse model...", flush=True)
coarse_model = train_coarse_model(x_train, y_coarse_train)

coarse_preds = coarse_model.predict(x_test)
coarse_report = classification_report(y_coarse_test, coarse_preds)

print("\n=== COARSE CLASSIFICATION REPORT ===", flush=True)
print(coarse_report, flush=True)

joblib.dump(coarse_model, MODEL_DIR / "coarse_model.joblib")


# ============================================================
# FINE-GRAINED MODELS
# ============================================================

def train_fine_models(df):
    models = {}

    for group in ["G", "F", "C"]:
        subset = df[df["POSITION_COARSE"] == group]

        x = subset.drop(columns=[c for c in DROP_COLS if c in subset.columns])
        y = subset["POSITION_FINE"]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            ))
        ])

        model.fit(x, y)
        models[group] = model

        print(f"Trained fine model for {group} ({len(subset)} samples)", flush=True)

    return models


fine_models = train_fine_models(train_df)
joblib.dump(fine_models, MODEL_DIR / "fine_models.joblib")


# ============================================================
# HIERARCHICAL PREDICTION
# ============================================================

def hierarchical_predict(x):
    coarse_preds = coarse_model.predict(x)
    final_preds = []

    for i, coarse in enumerate(coarse_preds):
        model = fine_models[coarse]
        final_preds.append(model.predict(x.iloc[[i]])[0])

    return np.array(final_preds)


# ============================================================
# FINAL EVALUATION
# ============================================================

final_preds = hierarchical_predict(x_test)
fine_report = classification_report(y_fine_test, final_preds)

print("\n=== 7-CLASS CLASSIFICATION REPORT ===", flush=True)
print(fine_report, flush=True)

# Save both reports to a single text file
report_path = RESULTS_DIR / "logistic_regression_report.txt"
with open(report_path, "w") as f:
    f.write("=== LOGISTIC REGRESSION — COARSE MODEL (G / F / C) ===\n\n")
    f.write(f"Test season: {TEST_SEASON}\n\n")
    f.write(str(coarse_report))
    f.write("\n\n=== LOGISTIC REGRESSION — 7-CLASS HIERARCHICAL MODEL ===\n\n")
    f.write(f"Test season: {TEST_SEASON}\n\n")
    f.write(str(fine_report))
print(f"Reports saved to {report_path}", flush=True)

# Confusion Matrix
labels = sorted(y_fine_test.unique())
cm = confusion_matrix(y_fine_test, final_preds, labels=labels)

disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("7-Class Confusion Matrix")
plt.tight_layout()

cm_path = RESULTS_DIR / "logistic_hierarchical_confusion_matrix.png"
plt.savefig(cm_path)
print(f"Confusion matrix saved to {cm_path}", flush=True)
plt.show()
plt.close()
