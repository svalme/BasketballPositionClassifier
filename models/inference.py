import joblib
import pandas as pd

# --------------------------------------------------
# Load models
# --------------------------------------------------

COARSE_MODEL_PATH = "models/coarse_model.joblib"
FINE_MODELS_PATH = "models/fine_models.joblib"

coarse_model = joblib.load(COARSE_MODEL_PATH)
fine_models = joblib.load(FINE_MODELS_PATH)


# --------------------------------------------------
# Feature list (must match training)
# --------------------------------------------------

FEATURES = [
    "PTS", "REB", "AST", "STL", "BLK",
    "PTS_PG", "REB_PG", "AST_PG", "STL_PG", "BLK_PG",
    "PTS_P36", "REB_P36", "AST_P36", "STL_P36", "BLK_P36",
    "AST_TO_TOV", "FG3_RATE", "FT_RATE"
]


# --------------------------------------------------
# Inference function
# --------------------------------------------------

def predict_position(player_stats: dict):
    """
    player_stats: dict with feature names as keys
    """

    df = pd.DataFrame([player_stats])

    # Ensure correct feature order
    X = df[FEATURES]

    # coarse prediction
    coarse_pred = coarse_model.predict(X)[0]

    # fine prediction
    fine_model = fine_models[coarse_pred]
    fine_pred = fine_model.predict(X)[0]

    return {
        "coarse_position": coarse_pred,
        "fine_position": fine_pred
    }
