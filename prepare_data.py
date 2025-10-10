# prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------
# 1. Load and Filter
# ------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Load raw NBA player stats from CSV."""
    return pd.read_csv(csv_path)


def filter_players(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only players with games played > 0."""
    return df[df['GP'] > 0].copy()

# ------------------------
# 2. Feature Engineering
# ------------------------
def add_per_game_stats(df: pd.DataFrame, stats: list) -> pd.DataFrame:
    """Add per-game stats to dataframe."""
    for stat in stats:
        df[f'{stat}_PG'] = df[stat] / df['GP']
    return df


def add_per36_stats(df: pd.DataFrame, stats: list) -> pd.DataFrame:
    """Add per-36-minute stats to dataframe."""
    for stat in stats:
        df[f'{stat}_P36'] = df[stat] / df['MIN'] * 36
    return df


def prepare_for_ml(df: pd.DataFrame, stats: list) -> pd.DataFrame:
    """Prepare dataframe with features and position labels for ML."""
    # Round for readability
    df = df.round(2)

    # Columns: identifiers + raw efficiency + engineered
    raw_efficiency = ["PLAYER_AGE", "FG_PCT", "FG3_PCT", "FT_PCT"]
    feature_cols = raw_efficiency \
                   + [f'{stat}_PG' for stat in stats] \
                   + [f'{stat}_P36' for stat in stats]

    ml_df = df[['PLAYER_ID', 'PLAYER', 'POSITION'] + feature_cols]
    return ml_df


def prepare_data(csv_path: str) -> pd.DataFrame:
    """Full pipeline: load, filter, engineer features, return ML-ready dataframe."""
    df = load_data(csv_path)

    per_game_stats = [
        'PTS','REB','AST','STL','BLK',
        'TOV','PF','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB'
    ]

    df = filter_players(df)
    df = add_per_game_stats(df, per_game_stats)
    df = add_per36_stats(df, per_game_stats)
    ml_df = prepare_for_ml(df, per_game_stats)

    return ml_df

def get_features_and_labels(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Returns train-test split ready for ML:
    X_train, X_test, y_train, y_test
    """
    df = prepare_data(csv_path)

    # Features = everything except identifiers + label
    X = df.drop(columns=['PLAYER_ID', 'PLAYER', 'POSITION'])
    y = df['POSITION']

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
