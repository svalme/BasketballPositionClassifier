import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# ============================================================
# PATHS
# ============================================================

ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data"   / "final_dataset.csv"
MODEL_PATH = ROOT / "models" / "mlp_coarse_model.joblib"
ASSETS_DIR = ROOT / "analysis" / "assets"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

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
    "POSITION_COARSE",
]

TEST_SEASON = "2024_25"

# Use only the held-out test season so importance scores aren't inflated
# by memorized training examples.
test_df = df[df["SEASON"] == TEST_SEASON].copy()
X = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])
y = test_df["POSITION_COARSE"]

# Cast to float so integer rank columns don't trigger a FutureWarning
# (and a hard ValueError in scikit-learn 1.9) during PDP computation.
X = X.astype(float)

# Load trained model
model = joblib.load(MODEL_PATH)

# ============================================================
# 1. PERMUTATION IMPORTANCE
# ============================================================

print("\n=== PERMUTATION IMPORTANCE ===")

perm = permutation_importance(
    model,
    X,
    y,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importances = pd.Series(perm.importances_mean, index=X.columns)
importances = importances.sort_values(ascending=False)

print(importances.head(15))

# Plot
plt.figure(figsize=(10, 6))
importances.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (Permutation)")
plt.xlabel("Decrease in accuracy")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "mlp_coarse_permutation_importance.png")
print(f"Saved: {ASSETS_DIR / 'mlp_coarse_permutation_importance.png'}")
plt.show()
plt.close()

# ============================================================
# 2. PARTIAL DEPENDENCE PLOTS
# ============================================================

POSITION_NAMES = {"G": "guard", "F": "forward", "C": "center"}

top_features = importances.head(4).index.tolist()
for cls in ["G", "F", "C"]:
    pos_name = POSITION_NAMES[cls]
    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=top_features,
        target=cls,
        ax=ax
    )

    plt.suptitle(f"Partial Dependence — {cls}")
    plt.tight_layout()
    fname = f"mlp_coarse_pdp_{pos_name}.png"
    plt.savefig(ASSETS_DIR / fname)
    print(f"Saved: {ASSETS_DIR / fname}")
    plt.show()
    plt.close()

# ============================================================
# 3. OPTIONAL: SHAP EXPLANATIONS
# ============================================================

try:
    import shap

    print("\nComputing SHAP values (this may take a moment)...")

    sample = X.sample(100, random_state=42)

    # shap.Explainer can't introspect a sklearn Pipeline directly.
    # KernelExplainer is model-agnostic: it only needs predict_proba.
    # kmeans(sample, 10) summarises the background into 10 points so
    # computation stays fast while still being representative.
    background  = shap.kmeans(sample, 10)
    # Wrap in a function that restores column names before calling the model.
    # SHAP strips column names when it builds perturbed samples (passing bare
    # numpy arrays), but StandardScaler was fitted with a named DataFrame, so
    # it warns on every call without them.  Re-wrapping as a DataFrame silences
    # the warning and keeps the feature order correct.
    feature_names = sample.columns.tolist()
    predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))
    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(sample)

    class_names = list(model.named_steps["mlp"].classes_)

    shap_path = ASSETS_DIR / "mlp_coarse_shap_summary.png"
    shap.summary_plot(shap_values, sample, class_names=class_names, show=False)
    plt.savefig(shap_path, bbox_inches="tight")
    print(f"Saved: {shap_path}")
    plt.show()
    plt.close()
except ImportError:
    print("SHAP not installed. Run `pip install shap` to enable SHAP plots.")
