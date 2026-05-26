import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

SEASONS = ["2020_21", "2021_22", "2022_23", "2023_24", "2024_25"]


def expand_wnba_positions(pos):
    """Convert WNBA abbreviated positions to full names."""
    mapping = {
        'G':   'Guard',
        'F':   'Forward',
        'C':   'Center',
        'G-F': 'Guard-Forward',
        'F-G': 'Forward-Guard',
        'F-C': 'Forward-Center',
        'C-F': 'Center-Forward',
    }
    return mapping.get(pos, pos)


def load_all_positions(league: str) -> pd.DataFrame:
    """Load and concatenate position files for all available seasons."""
    frames = []
    for season in SEASONS:
        path = DATA_DIR / season / f"{league}_{season}_positions.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["SEASON"] = season
            frames.append(df)
        else:
            print(f"  Skipped (not found): {path.name}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================
# NBA
# ============================================================

print("NBA Positions:")
nba_df = load_all_positions("nba")
nba_df = nba_df.dropna(subset=["POSITION"])

nba_possible_positions = set(nba_df["POSITION"].unique())
print(sorted(nba_possible_positions))
print("NBA Positions:\n", nba_df["POSITION"].value_counts())

nba_duplicates = nba_df[nba_df["PLAYER_ID"].duplicated(keep=False)]
if len(nba_duplicates) > 0:
    print(f"\nNBA has {len(nba_duplicates)} duplicate IDs (same player, multiple seasons — expected):")
    print(nba_duplicates[["PLAYER_ID", "POSITION", "SEASON"]].head(10))
else:
    print(f"\nNBA: All {len(nba_df)} rows have unique PLAYER_ID")

# ============================================================
# WNBA
# ============================================================

print("\nWNBA Positions:")
wnba_df = load_all_positions("wnba")

wnba_df_expanded = wnba_df.copy()
wnba_df_expanded["POSITION"] = wnba_df_expanded["POSITION"].apply(expand_wnba_positions)

new_wnba_possible_positions = set(wnba_df_expanded["POSITION"].unique())
print(sorted(new_wnba_possible_positions))
print("WNBA Positions:\n", wnba_df_expanded["POSITION"].value_counts())

wnba_duplicates = wnba_df[wnba_df["PLAYER_ID"].duplicated(keep=False)]
if len(wnba_duplicates) > 0:
    print(f"\nWNBA has {len(wnba_duplicates)} duplicate IDs (same player, multiple seasons — expected):")
    print(wnba_duplicates[["PLAYER_ID", "POSITION", "SEASON"]].head(10))
else:
    print(f"\nWNBA: All {len(wnba_df)} rows have unique PLAYER_ID")

# ============================================================
# Cross-league overlap check (IDs should never overlap)
# ============================================================

print("\n" + "=" * 50)
print("Cross-League Check:")
print("=" * 50)

nba_ids  = set(nba_df["PLAYER_ID"].unique())
wnba_ids = set(wnba_df["PLAYER_ID"].unique())

overlap = nba_ids & wnba_ids
if overlap:
    print(f"\n{len(overlap)} player IDs appear in BOTH NBA and WNBA:")
    print(overlap)
else:
    print(f"\nNo overlap: NBA and WNBA have completely different player IDs")

print(f"\nNBA unique player IDs:  {len(nba_ids)}")
print(f"WNBA unique player IDs: {len(wnba_ids)}")
print(f"Total unique players across both leagues: {len(nba_ids | wnba_ids)}")
