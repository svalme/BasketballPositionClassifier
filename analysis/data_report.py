import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / "data"     / "final_dataset.csv"
ASSETS_DIR  = ROOT / "analysis" / "assets"
RESULTS_DIR = ROOT / "results"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TEST_SEASON  = "2024_25"
FINE_ORDER   = ["Guard", "Guard-Forward", "Forward",
                "Forward-Guard", "Forward-Center", "Center", "Center-Forward"]
COARSE_ORDER = ["G", "F", "C"]
COARSE_NAMES = {"G": "Guard", "F": "Forward", "C": "Center"}

# ============================================================
# LOAD
# ============================================================

df       = pd.read_csv(DATA_PATH)
train_df = df[df["SEASON"] != TEST_SEASON].copy()
test_df  = df[df["SEASON"] == TEST_SEASON].copy()

# ============================================================
# TEXT REPORT HELPERS
# ============================================================

lines = []

def section(title):
    lines.append("")
    lines.append("=" * 65)
    lines.append(f"  {title}")
    lines.append("=" * 65)

def row(s=""):
    lines.append(s)


# ============================================================
# SECTION 1 — DATASET OVERVIEW
# ============================================================

section("DATASET OVERVIEW")
row(f"  Total player-seasons : {len(df)}")
row(f"  Train seasons        : {', '.join(sorted(df[df['SEASON'] != TEST_SEASON]['SEASON'].unique()))}")
row(f"  Test season          : {TEST_SEASON}")
row(f"  Train rows           : {len(train_df)}")
row(f"  Test rows            : {len(test_df)}")

nba  = df[df["LEAGUE"] == "nba"]
wnba = df[df["LEAGUE"] == "wnba"]
row("")
row(f"  NBA  rows : {len(nba):>5}  ({len(nba)  / len(df) * 100:.1f}%)")
row(f"  WNBA rows : {len(wnba):>5}  ({len(wnba) / len(df) * 100:.1f}%)")


# ============================================================
# SECTION 2 — COARSE DISTRIBUTION
# ============================================================

section("COARSE POSITION DISTRIBUTION  (G / F / C)")
row(f"  {'Position':<14} {'Total':>7} {'% total':>8} {'Train':>7} {'Test':>7}")
row("  " + "-" * 47)

for pos in COARSE_ORDER:
    total   = (df["POSITION_COARSE"]       == pos).sum()
    train_n = (train_df["POSITION_COARSE"] == pos).sum()
    test_n  = (test_df["POSITION_COARSE"]  == pos).sum()
    row(f"  {COARSE_NAMES[pos]:<14} {total:>7} {total/len(df)*100:>7.1f}% {train_n:>7} {test_n:>7}")


# ============================================================
# SECTION 3 — FINE-GRAINED DISTRIBUTION
# ============================================================

section("FINE-GRAINED POSITION DISTRIBUTION")
row(f"  {'Position':<18} {'Total':>7} {'% total':>8} {'Train':>7} {'Test':>7}  {'Note'}")
row("  " + "-" * 63)

for pos in FINE_ORDER:
    total   = (df["POSITION_FINE"]       == pos).sum()
    train_n = (train_df["POSITION_FINE"] == pos).sum()
    test_n  = (test_df["POSITION_FINE"]  == pos).sum()
    note    = " ← thin (<30 test examples)" if test_n < 30 else ""
    row(f"  {pos:<18} {total:>7} {total/len(df)*100:>7.1f}% {train_n:>7} {test_n:>7}  {note}")

row("")
row("  Positions flagged 'thin' have fewer than 30 test examples.")
row("  F1 scores for these classes are less statistically reliable.")
row("  Collecting more data for hybrid roles would most improve the")
row("  fine-grained model.")


# ============================================================
# SECTION 4 — BY LEAGUE
# ============================================================

section("FINE-GRAINED DISTRIBUTION BY LEAGUE")

for league in ["nba", "wnba"]:
    sub = df[df["LEAGUE"] == league]
    row(f"\n  {league.upper()}  ({len(sub)} rows)")
    row(f"  {'Position':<18} {'Count':>7}  {'%':>6}")
    row("  " + "-" * 35)
    for pos in FINE_ORDER:
        n = (sub["POSITION_FINE"] == pos).sum()
        row(f"  {pos:<18} {n:>7}  {n/len(sub)*100:>5.1f}%")


# ============================================================
# SECTION 5 — BY SEASON
# ============================================================

section("ROWS PER SEASON AND LEAGUE")
pivot = df.groupby(["SEASON", "LEAGUE"]).size().unstack(fill_value=0)
pivot["TOTAL"] = pivot.sum(axis=1)
row("")
row("  " + pivot.to_string().replace("\n", "\n  "))


# ============================================================
# SECTION 6 — AGE BY COARSE POSITION
# ============================================================

section("AGE BY COARSE POSITION")
row(f"  {'Position':<14} {'Mean':>6} {'Median':>8} {'Min':>6} {'Max':>6} {'Std':>6}")
row("  " + "-" * 48)

for pos in COARSE_ORDER:
    ages = df[df["POSITION_COARSE"] == pos]["AGE"].dropna()
    row(f"  {COARSE_NAMES[pos]:<14} {ages.mean():>6.1f} {ages.median():>8.1f}"
        f" {ages.min():>6.0f} {ages.max():>6.0f} {ages.std():>6.1f}")


# ============================================================
# SECTION 7 — STAT PROFILES BY COARSE POSITION
# ============================================================

section("MEAN STAT PROFILES BY COARSE POSITION")

KEY_STATS = ["PTS", "REB", "AST", "BLK", "STL",
             "FG3_PCT", "FG_PCT", "AST_P36", "BLK_P36", "REB_P36"]
KEY_STATS = [s for s in KEY_STATS if s in df.columns]

row(f"\n  {'Stat':<14} {'Guard':>9} {'Forward':>9} {'Center':>9}  {'Highest'}")
row("  " + "-" * 55)

for stat in KEY_STATS:
    vals    = {p: df[df["POSITION_COARSE"] == p][stat].mean() for p in COARSE_ORDER}
    highest = COARSE_NAMES[max(vals, key=vals.get)]
    row(f"  {stat:<14} {vals['G']:>9.2f} {vals['F']:>9.2f} {vals['C']:>9.2f}  {highest}")

row("")
row("  Read as: who has the highest average value for each stat.")
row("  FG3_PCT = 3-point %, FG_PCT = field goal %, AST_P36 = assists per 36 min.")


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = RESULTS_DIR / "data_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Report saved  →  {report_path}", flush=True)


# ============================================================
# CHART 1 — Fine-grained class distribution: train vs test
# ============================================================

fig, ax = plt.subplots(figsize=(13, 6))
x     = np.arange(len(FINE_ORDER))
width = 0.35

train_counts = [(train_df["POSITION_FINE"] == p).sum() for p in FINE_ORDER]
test_counts  = [(test_df["POSITION_FINE"]  == p).sum() for p in FINE_ORDER]

bars_train = ax.bar(x - width / 2, train_counts, width, label="Train", color="steelblue")
bars_test  = ax.bar(x + width / 2, test_counts,  width, label="Test",  color="coral")

# Count labels on top of each bar
for bar in bars_train:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
for bar in bars_test:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(FINE_ORDER, rotation=25, ha="right")
ax.set_ylabel("Number of player-seasons")
ax.set_title("Fine-Grained Position Distribution — Train vs Test")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

path = ASSETS_DIR / "data_class_distribution.png"
plt.savefig(path)
print(f"Saved  →  {path}", flush=True)
plt.show()
plt.close()


# ============================================================
# CHART 2 — Age distribution by coarse position (box plot)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

age_data  = [df[df["POSITION_COARSE"] == p]["AGE"].dropna().values for p in COARSE_ORDER]
pos_labels = [COARSE_NAMES[p] for p in COARSE_ORDER]
colors     = ["#4C72B0", "#DD8452", "#55A868"]

bp = ax.boxplot(age_data, labels=pos_labels, patch_artist=True, notch=False)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color("black")
    median.set_linewidth(2)

ax.set_ylabel("Age")
ax.set_title("Age Distribution by Coarse Position")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

path = ASSETS_DIR / "data_age_by_position.png"
plt.savefig(path)
print(f"Saved  →  {path}", flush=True)
plt.show()
plt.close()


# ============================================================
# CHART 3 — Stat profiles by position (normalised)
# ============================================================

PROFILE_STATS = ["PTS", "REB", "AST", "BLK", "STL",
                 "FG3_PCT", "FG_PCT", "AST_P36", "BLK_P36"]
PROFILE_STATS = [s for s in PROFILE_STATS if s in df.columns]

means = pd.DataFrame(
    {p: df[df["POSITION_COARSE"] == p][PROFILE_STATS].mean() for p in COARSE_ORDER}
)

# Normalise each stat so the highest value across positions = 1.0
# This makes it easy to see which position leads each category.
means_norm = means.div(means.max(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(13, 6))
x      = np.arange(len(PROFILE_STATS))
width  = 0.25
colors = ["#4C72B0", "#DD8452", "#55A868"]

for i, (pos, color) in enumerate(zip(COARSE_ORDER, colors)):
    ax.bar(x + i * width, means_norm[pos], width,
           label=COARSE_NAMES[pos], color=color, alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(PROFILE_STATS, rotation=20, ha="right")
ax.set_ylabel("Relative value  (1.0 = highest among the three positions)")
ax.set_title("Stat Profiles by Position  —  normalised for comparison")
ax.legend()
ax.set_ylim(0, 1.2)
ax.grid(axis="y", alpha=0.3)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()

path = ASSETS_DIR / "data_stat_profiles.png"
plt.savefig(path)
print(f"Saved  →  {path}", flush=True)
plt.show()
plt.close()

print("\nAll done.", flush=True)
