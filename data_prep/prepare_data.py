# prepare_data.py
import os
import pandas as pd
from pathlib import Path

BASE_DIR = "data"
SEASONS = ["2020_21", "2021_22", "2022_23", "2023_24", "2024_25"]
LEAGUES = ["nba", "wnba"]

DROP_COLUMNS = ["PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]


def load_season_league(season, league):
    """Load stats + positions for one league and season."""
    root = Path(__file__).parent.parent
    season_dir = root / BASE_DIR / season

    stats_path = season_dir / f"{league}_{season}_stats.csv"
    pos_path = season_dir / f"{league}_{season}_positions.csv"

    stats = pd.read_csv(stats_path)
    pos = pd.read_csv(pos_path)

    df = stats.merge(pos, on="PLAYER_ID", how="inner")
    df["SEASON"] = season
    df["LEAGUE"] = league

    return df


def load_all_data():
    """Load all seasons and leagues, skipping missing files."""
    dfs = []
    for season in SEASONS:
        for league in LEAGUES:
            try:
                print(f"Loading {league.upper()} {season}")
                dfs.append(load_season_league(season, league))
            except FileNotFoundError as e:
                print(f"     Skipped: {e}")
            except Exception as e:
                print(f"    Error loading {league.upper()} {season}: {e}")
    
    if not dfs:
        raise ValueError("No data files found. Check your SEASONS and file paths.")
    
    return pd.concat(dfs, ignore_index=True)



def add_features(df):
    # remove players who didn't play
    df = df[df["GP"] > 0].copy()

    # per-game stats
    for stat in ["PTS", "REB", "AST", "STL", "BLK"]:
        df[f"{stat}_PG"] = df[stat] / df["GP"]

    # per-36-minute stats
    for stat in ["PTS", "REB", "AST", "STL", "BLK"]:
        df[f"{stat}_P36"] = (df[stat] / df["MIN"]) * 36

    # ratio features (with division by zero protection)
    df["AST_TO_TOV"] = df["AST"] / (df["TOV"] + 1e-6)
    df["FG3_RATE"] = df["FG3A"] / (df["FGA"] + 1e-6)  # avoid division by zero
    df["FT_RATE"] = df["FTA"] / (df["FGA"] + 1e-6)  # avoid division by zero

    return df



def normalize_position(pos):
    """Convert detailed position into coarse class."""
    if not isinstance(pos, str):
        return None
    pos = pos.strip()
    if pos.startswith("G"):
        return "G"
    elif pos.startswith("F"):
        return "F"
    elif pos.startswith("C"):
        return "C"
    return None

FINE_POSITION_MAP = {
    # guards
    "G": "Guard",
    "Guard": "Guard",
    "G-F": "Guard-Forward",
    "Guard-Forward": "Guard-Forward",

    # forwards
    "F": "Forward",
    "Forward": "Forward",
    "F-G": "Forward-Guard",
    "Forward-Guard": "Forward-Guard",
    "F-C": "Forward-Center",
    "Forward-Center": "Forward-Center",

    # centers
    "C": "Center",
    "Center": "Center",
    "C-F": "Center-Forward",
    "Center-Forward": "Center-Forward",
}

def normalize_fine_position(pos):
    return FINE_POSITION_MAP.get(pos, None)

def add_labels(df):
    df = df.copy()  # ensure we're working with a copy, not a view
    df["POSITION_FINE"] = df["POSITION"].apply(normalize_fine_position)
    df = df.dropna(subset=["POSITION_FINE"]).copy()
    df["POSITION_COARSE"] = df["POSITION"].apply(normalize_position)
    df = df.dropna(subset=["POSITION_COARSE", "POSITION"]).copy()
    return df



def prepare_dataset():
    df = load_all_data()
    df = add_features(df)
    df = add_labels(df)

    # drop unused columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # fill any NaN values (from division by zero protection) with 0
    df = df.fillna(0)
    
    # drop rows where position labels are None (invalid positions)
    df = df[df["POSITION_COARSE"].notna()]

    return df


if __name__ == "__main__":  
    df = prepare_dataset()

    print("Final dataset shape:", df.shape)
    print("\nLabel distribution (coarse):")
    print(df["POSITION_COARSE"].value_counts())

    print("\nLabel distribution (fine):")
    print(df["POSITION_FINE"].value_counts())

    output_path = "../data/final_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved final dataset to {output_path}")
  
    