import json
import joblib
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).parent.parent
COARSE_MODEL_PATH = ROOT / "models" / "coarse_model.joblib"
FINE_MODELS_PATH  = ROOT / "models" / "fine_models.joblib"
FEATURES_PATH     = ROOT / "models" / "feature_columns.json"

# ============================================================
# Load models and feature list saved during training
# ============================================================

coarse_model = joblib.load(COARSE_MODEL_PATH)
fine_models  = joblib.load(FINE_MODELS_PATH)

with open(FEATURES_PATH) as f:
    FEATURES = json.load(f)


# ============================================================
# Inference function
# ============================================================

def predict_position(player_stats: dict) -> dict:
    """
    Predict the coarse (G/F/C) and fine-grained position for a player.

    Args:
        player_stats: dict mapping feature names to values.
                      Missing features default to 0; extra keys are ignored.

    Returns:
        dict with keys 'coarse_position' and 'fine_position'.
    """
    df = pd.DataFrame([player_stats])

    # Align to the exact feature columns seen during training;
    # fill any missing columns with 0 rather than raising KeyError.
    X = df.reindex(columns=FEATURES, fill_value=0)

    coarse_pred = coarse_model.predict(X)[0]

    fine_model = fine_models[coarse_pred]
    fine_pred  = fine_model.predict(X)[0]

    return {
        "coarse_position": coarse_pred,
        "fine_position":   fine_pred,
    }
