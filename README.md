# Basketball Position Classifier

A machine learning system that classifies basketball player positions from multi-season NBA and WNBA performance data. Rather than relying on fixed position labels, the models learn positional roles directly from statistical profiles — reflecting how fluid and role-based modern basketball has become.

---

## How it works

The system uses a two-stage hierarchical approach:

**Stage 1 — Coarse classifier** predicts one of three broad roles:

| Label | Role |
|---|---|
| G | Guard — primary ball-handlers and perimeter players |
| F | Forward — wings and versatile scorers/defenders |
| C | Center — interior players focused on rebounding and rim protection |

**Stage 2 — Fine-grained classifier** takes the coarse prediction and routes the player to a specialised model trained only on that group:

| Coarse | Fine-grained options |
|---|---|
| Guard | Guard, Guard-Forward |
| Forward | Forward, Forward-Guard, Forward-Center |
| Center | Center, Center-Forward |

Each fine classifier only sees examples from its own group, which reduces ambiguity and reflects real positional distinctions in modern basketball.

---

## Fine-grained positions

| Label | Description |
|---|---|
| Guard | Traditional point or shooting guard |
| Guard-Forward | Combo guard / wing scorer |
| Forward | Standard wing or power forward |
| Forward-Guard | Wing with strong ball-handling responsibilities |
| Center | Traditional interior big |
| Center-Forward | Stretch big or mobile center with perimeter ability |
| Forward-Center | Hybrid big-forward role |

---

## Dataset

- **Source:** NBA API (`nba_api` Python package)
- **Leagues:** NBA and WNBA
- **Seasons:** 2020-21 through 2024-25 (five seasons)
- **Train / test split:** seasons 2020-21 to 2023-24 for training, 2024-25 held out for testing
- **Size:** ~3,000 player-seasons across both leagues

Position labels are normalised from raw API strings (e.g. `"G-F"` → `"Guard-Forward"`) and mapped to both coarse and fine-grained categories. Players with zero games played are excluded.

---

## Features

Each player is represented by stats from the API (per-game averages and rate stats):

- **Raw per-game stats:** PTS, REB, AST, STL, BLK, FG%, 3P%, FT%
- **Per-36-minute stats:** PTS_P36, REB_P36, AST_P36, STL_P36, BLK_P36
- **Ratio features:** AST-to-turnover ratio, 3-point attempt rate, free-throw rate
- **League ranking columns** for all major stat categories

---

## Models

Two models are trained and compared:

**Logistic Regression (hierarchical)**
- Stage 1: coarse 3-class logistic regression
- Stage 2: three separate logistic regressions, one per coarse group
- Uses `StandardScaler` normalisation and balanced class weights

**Neural Network / MLP (coarse only)**
- Two hidden layers (32 → 16 units), ReLU activation
- Trained on the same features and split as the logistic regression coarse model
- Used for model interpretation (permutation importance, PDPs, SHAP)

---

## Results

Test season: 2024-25 (held out, never seen during training)

**Coarse model — 3 classes (G / F / C)**

| Model | Accuracy | Guard F1 | Forward F1 | Center F1 |
|---|---|---|---|---|
| Logistic Regression | 76% | 0.86 | 0.69 | 0.65 |
| Neural Network | 76% | 0.83 | 0.70 | 0.71 |

Both models reach the same overall accuracy. The MLP is more precise on Centers; logistic regression is stronger on Guards.

**Hierarchical model — 7 classes**

| Position | F1 | Test examples |
|---|---|---|
| Guard | 0.79 | 238 |
| Forward | 0.57 | 183 |
| Center | 0.56 | 62 |
| Guard-Forward | 0.23 | 39 |
| Forward-Center | 0.25 | 33 |
| Center-Forward | 0.25 | 20 |
| Forward-Guard | 0.26 | 14 |
| **Overall** | **0.58** | **589** |

Pure positions classify well. Hybrid positions (Guard-Forward, Forward-Center, etc.) are harder — they are statistically ambiguous by design, and some have very few training examples. More labelled data for hybrid roles would be the most impactful improvement.

**Top predictive features** (permutation importance on MLP):
3-point shooting percentage is the single strongest positional signal in the modern game, followed by assists per 36 minutes and defensive rebounding.

---

## Project structure

```
BasketballPositionClassifier/
├── data/
│   ├── 2020_21/ … 2024_25/          raw per-season CSVs (stats + positions)
│   ├── final_dataset.csv            combined, feature-engineered dataset
│   └── analyze_data.py              raw data diagnostics
├── data_prep/
│   ├── fetch_stats.py               pulls data from NBA API
│   └── prepare_data.py              builds final_dataset.csv
├── models/
│   ├── train_model.py               trains logistic regression (hierarchical)
│   ├── train_nn.py                  trains neural network (coarse)
│   ├── inference.py                 predict position for a new player
│   ├── coarse_model.joblib
│   ├── fine_models.joblib
│   ├── mlp_coarse_model.joblib
│   └── feature_columns.json        feature list saved at training time
├── analysis/
│   ├── data_report.py               class distributions, age, stat profiles
│   ├── interpret_mlp_coarse_model.py  feature importance, PDPs, SHAP
│   ├── visualize_decision_boundary.py
│   └── assets/                      saved charts
├── results/                         saved classification reports + confusion matrices
├── tests/
│   └── test_prepare_data.py
├── requirements.txt
└── README.md
```

---

## Installation

```
pip install -r requirements.txt
```

---

## Running inference

`inference.py` is a module — import it rather than running it directly:

```python
from models.inference import predict_position

result = predict_position({
    "PTS": 18.4,
    "REB": 5.1,
    "AST": 6.2,
    "STL": 1.3,
    "BLK": 0.4,
    "MIN": 32.0,
    # any other stats — missing features default to 0
})

print(result)
# {"coarse_position": "G", "fine_position": "Guard-Forward"}
```

The models must be trained before calling this (run `train_model.py` first).
