import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / "data"    / "final_dataset.csv"
MODEL_DIR   = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TEST_SEASON = "2024_25"

# ============================================================
# CONFIG
# ============================================================

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

print("Loading data...", flush=True)
df = pd.read_csv(DATA_PATH)

train_df = df[df["SEASON"] != TEST_SEASON].copy()
test_df  = df[df["SEASON"] == TEST_SEASON].copy()

X_train = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
X_test  = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])

y_train = train_df["POSITION_COARSE"]
y_test  = test_df["POSITION_COARSE"]

print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}", flush=True)

# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=1000,
            random_state=42
        ))
    ])
    model.fit(X, y)
    return model


print("\nTraining neural network...", flush=True)
model = train_model(X_train, y_train)
print("Training complete.", flush=True)

# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print("\n=== CLASSIFICATION REPORT ===", flush=True)
print(report, flush=True)

# Save report to file
report_path = RESULTS_DIR / "neural_network_report.txt"
with open(report_path, "w") as f:
    f.write("=== NEURAL NETWORK — COARSE MODEL (G / F / C) ===\n\n")
    f.write(f"Test season: {TEST_SEASON}\n\n")
    f.write(str(report))
print(f"Report saved to {report_path}", flush=True)

# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("3-Class Confusion Matrix (Neural Network)")
plt.tight_layout()

cm_path = RESULTS_DIR / "mlp_coarse_confusion_matrix.png"
plt.savefig(cm_path)
print(f"Confusion matrix saved to {cm_path}", flush=True)
plt.show()
plt.close()

# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "mlp_coarse_model.joblib"
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}", flush=True)
