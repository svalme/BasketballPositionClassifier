import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load data and model
# -------------------------------

ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data"   / "final_dataset.csv"
MODEL_PATH = ROOT / "models" / "coarse_model.joblib"
ASSETS_DIR = ROOT / "analysis" / "assets"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[
    "PLAYER_ID", "NICKNAME", "LEAGUE",
    "SEASON", "POSITION", "POSITION_COARSE", "POSITION_FINE"
])

y = df["POSITION_COARSE"]

model = joblib.load(MODEL_PATH)

# -------------------------------
# PCA for visualization
# -------------------------------

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
x_2d = pca.fit_transform(x_scaled)

# -------------------------------
# Train visualization model
# -------------------------------

from sklearn.linear_model import LogisticRegression

viz_model = LogisticRegression(max_iter=2000)
viz_model.fit(x_2d, y)

# -------------------------------
# Create mesh
# -------------------------------

x_min, x_max = x_2d[:, 0].min() - 1, x_2d[:, 0].max() + 1
y_min, y_max = x_2d[:, 1].min() - 1, x_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

#Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)

# Encode labels to integers
# Encode labels
y_encoded = y.astype("category").cat.codes
label_names = y.astype("category").cat.categories
label_map = {i: label for i, label in enumerate(label_names)}

# Predict decision regions
Z_labels = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array([label_map.keys().__iter__().__next__() for _ in Z_labels])  # placeholder
Z = np.array([list(label_map.keys())[list(label_map.values()).index(z)] for z in Z_labels])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="tab10")

scatter = plt.scatter(
    x_2d[:, 0],
    x_2d[:, 1],
    c=y_encoded,
    cmap="tab10",
    edgecolor="k",
    s=40
)

handles, _ = scatter.legend_elements()
plt.legend(handles, label_names, title="Position")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Decision Boundaries (Logistic Regression)")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "logistic_coarse_pca_decision_boundary.png")
print(f"Saved: {ASSETS_DIR / 'logistic_coarse_pca_decision_boundary.png'}")
plt.show()
plt.close()