import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_data import (
    normalize_position,
    normalize_fine_position,
    load_season_league,
    add_features,
    add_labels,
)


@pytest.fixture
def sample_stats_df():
    """Create a sample stats dataframe."""
    return pd.DataFrame({
        'PLAYER_ID': [1, 2, 3, 4, 5],
        'PLAYER_NAME': ['Player A', 'Player B', 'Player C', 'Player D', 'Player E'],
        'TEAM_ID': [101, 102, 101, 103, 102],
        'TEAM_ABBREVIATION': ['LAL', 'GSW', 'LAL', 'BOS', 'GSW'],
        'GP': [82, 70, 0, 50, 65],
        'MIN': [3000, 2500, 0, 1800, 2200],
        'PTS': [2000, 1800, 0, 1200, 1500],
        'REB': [400, 500, 0, 800, 600],
        'AST': [600, 300, 0, 100, 250],
        'STL': [150, 120, 0, 80, 100],
        'BLK': [50, 100, 0, 200, 80],
        'TOV': [200, 180, 0, 90, 120],
        'FGA': [1600, 1500, 0, 1000, 1200],
        'FG3A': [600, 700, 0, 300, 400],
        'FTA': [500, 400, 0, 400, 300],
    })


@pytest.fixture
def sample_positions_df():
    """Create a sample positions dataframe."""
    return pd.DataFrame({
        'PLAYER_ID': [1, 2, 3, 4, 5],
        'POSITION': ['Guard', 'Forward', 'Center', 'Guard-Forward', 'Forward-Center'],
    })


@pytest.fixture
def temp_season_dir():
    """Create a temporary season directory with CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        season_dir = Path(tmpdir) / "2023_24"
        season_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample data
        stats = pd.DataFrame({
            'PLAYER_ID': [1, 2, 3],
            'PLAYER_NAME': ['Player A', 'Player B', 'Player C'],
            'TEAM_ID': [101, 102, 101],
            'TEAM_ABBREVIATION': ['LAL', 'GSW', 'LAL'],
            'GP': [82, 70, 50],
            'MIN': [3000, 2500, 1800],
            'PTS': [2000, 1800, 1200],
            'REB': [400, 500, 800],
            'AST': [600, 300, 100],
            'STL': [150, 120, 80],
            'BLK': [50, 100, 200],
            'TOV': [200, 180, 90],
            'FGA': [1600, 1500, 1000],
            'FG3A': [600, 700, 300],
            'FTA': [500, 400, 400],
        })
        
        positions = pd.DataFrame({
            'PLAYER_ID': [1, 2, 3],
            'POSITION': ['Guard', 'Forward', 'Center'],
        })
        
        stats.to_csv(season_dir / "nba_2023_24_stats.csv", index=False)
        positions.to_csv(season_dir / "nba_2023_24_positions.csv", index=False)
        
        yield tmpdir, season_dir


class TestNormalizePosition:
    def test_guard_position(self):
        """Test that guard positions are normalized to 'G'."""
        assert normalize_position("Guard") == "G"
        assert normalize_position("Guard-Forward") == "G"
        assert normalize_position("G") == "G"
        assert normalize_position("G-F") == "G"

    def test_forward_position(self):
        """Test that forward positions are normalized to 'F'."""
        assert normalize_position("Forward") == "F"
        assert normalize_position("Forward-Guard") == "F"
        assert normalize_position("Forward-Center") == "F"
        assert normalize_position("F") == "F"

    def test_center_position(self):
        """Test that center positions are normalized to 'C'."""
        assert normalize_position("Center") == "C"
        assert normalize_position("Center-Forward") == "C"
        assert normalize_position("C") == "C"

    def test_invalid_position(self):
        """Test that invalid positions return None."""
        assert normalize_position("Invalid") is None
        assert normalize_position("") is None

    def test_non_string_position(self):
        """Test that non-string positions return None."""
        assert normalize_position(None) is None
        assert normalize_position(123) is None
        assert normalize_position(np.nan) is None

    def test_position_with_whitespace(self):
        """Test that positions with leading/trailing whitespace are handled."""
        assert normalize_position("  Guard  ") == "G"
        assert normalize_position(" Forward ") == "F"


class TestNormalizeFinePosition:
    """Test the fine-grained position normalization."""
    
    def test_guard_fine_positions(self):
        """Test guard fine-grained positions."""
        assert normalize_fine_position("Guard") == "Guard"
        assert normalize_fine_position("G") == "Guard"
        assert normalize_fine_position("Guard-Forward") == "Guard-Forward"
        assert normalize_fine_position("G-F") == "Guard-Forward"

    def test_forward_fine_positions(self):
        """Test forward fine-grained positions."""
        assert normalize_fine_position("Forward") == "Forward"
        assert normalize_fine_position("F") == "Forward"
        assert normalize_fine_position("Forward-Guard") == "Forward-Guard"
        assert normalize_fine_position("F-G") == "Forward-Guard"
        assert normalize_fine_position("Forward-Center") == "Forward-Center"
        assert normalize_fine_position("F-C") == "Forward-Center"

    def test_center_fine_positions(self):
        """Test center fine-grained positions."""
        assert normalize_fine_position("Center") == "Center"
        assert normalize_fine_position("C") == "Center"
        assert normalize_fine_position("Center-Forward") == "Center-Forward"
        assert normalize_fine_position("C-F") == "Center-Forward"

    def test_unmapped_position(self):
        """Test that unmapped positions return None."""
        assert normalize_fine_position("Invalid") is None
        assert normalize_fine_position("") is None
        assert normalize_fine_position(None) is None


class TestAddFeatures:
    def test_filters_zero_gp_players(self, sample_stats_df):
        """Test that players with GP=0 are filtered out."""
        result = add_features(sample_stats_df)
        assert len(result) == 4
        assert all(result['GP'] > 0)
        assert 'Player C' not in result['PLAYER_NAME'].values

    def test_adds_per_game_stats(self, sample_stats_df):
        """Test that per-game stats are calculated correctly."""
        result = add_features(sample_stats_df)
        
        assert 'PTS_PG' in result.columns
        assert 'REB_PG' in result.columns
        assert 'AST_PG' in result.columns
        
        # Player A: 2000 PTS / 82 GP ≈ 24.39
        assert abs(result.loc[0, 'PTS_PG'] - (2000 / 82)) < 0.01

    def test_adds_per36_stats(self, sample_stats_df):
        """Test that per-36-minute stats are calculated correctly."""
        result = add_features(sample_stats_df)
        
        assert 'PTS_P36' in result.columns
        assert 'REB_P36' in result.columns
        
        # Player A: 2000 PTS / 3000 MIN = 24.0 per 36
        expected = (2000 / 3000) * 36
        assert result.loc[0, 'PTS_P36'] == pytest.approx(expected, abs=1.0)  # Allow 1 point tolerance

    def test_adds_ratio_features(self, sample_stats_df):
        """Test that ratio features are calculated."""
        result = add_features(sample_stats_df)
        
        assert 'AST_TO_TOV' in result.columns
        assert 'FG3_RATE' in result.columns
        assert 'FT_RATE' in result.columns

    def test_handles_division_by_zero(self, sample_stats_df):
        """Test that division by zero is handled gracefully."""
        sample_stats_df.loc[0, 'TOV'] = 0
        result = add_features(sample_stats_df)
        
        # AST_TO_TOV should not be NaN or inf
        assert not np.isinf(result.loc[0, 'AST_TO_TOV'])


class TestAddLabels:
    def test_adds_fine_position(self, sample_stats_df, sample_positions_df):
        """Test that fine-grained position labels are added."""
        df = sample_stats_df.merge(sample_positions_df, on='PLAYER_ID')
        result = add_labels(df)
        
        assert 'POSITION_FINE' in result.columns
        # Player 1 has position 'Guard' which maps to 'Guard'
        assert result.iloc[0]['POSITION_FINE'] == 'Guard'

    def test_adds_coarse_position(self, sample_stats_df, sample_positions_df):
        """Test that coarse position labels are added."""
        df = sample_stats_df.merge(sample_positions_df, on='PLAYER_ID')
        result = add_labels(df)
        
        assert 'POSITION_COARSE' in result.columns
        # Player 1: Guard → G
        assert result.iloc[0]['POSITION_COARSE'] == 'G'

    def test_handles_combined_positions_fine(self, sample_stats_df, sample_positions_df):
        """Test that combined positions are mapped correctly in fine labels."""
        df = sample_stats_df.merge(sample_positions_df, on='PLAYER_ID')
        result = add_labels(df)
        
        # Player 4: Guard-Forward → Guard-Forward
        assert result.iloc[3]['POSITION_FINE'] == 'Guard-Forward'
        # Player 5: Forward-Center → Forward-Center
        assert result.iloc[4]['POSITION_FINE'] == 'Forward-Center'

    def test_handles_combined_positions_coarse(self, sample_stats_df, sample_positions_df):
        """Test that combined positions are coarsened correctly."""
        df = sample_stats_df.merge(sample_positions_df, on='PLAYER_ID')
        result = add_labels(df)
        
        # Guard-Forward should coarsen to G
        assert result.iloc[3]['POSITION_COARSE'] == 'G'
        # Forward-Center should coarsen to F
        assert result.iloc[4]['POSITION_COARSE'] == 'F'

    def test_dropna_removes_invalid_positions(self, sample_stats_df):
        """Test that rows with unmapped positions are removed."""
        # Create a dataframe with an invalid position
        df = sample_stats_df.merge(
            pd.DataFrame({
                'PLAYER_ID': [1, 2, 3, 4, 5],
                'POSITION': ['Guard', 'Forward', 'INVALID', 'Guard-Forward', 'Forward-Center']
            }),
            on='PLAYER_ID'
        )
        
        result = add_labels(df)
        
        # Player 3 with 'INVALID' should be dropped
        assert 'INVALID' not in result['POSITION'].values
        assert len(result) == 4  # Down from 5


class TestLoadSeasonLeague:
    def test_loads_stats_and_positions(self, temp_season_dir):
        """Test that stats and positions are loaded and merged."""
        tmpdir, season_dir = temp_season_dir
        
        # Temporarily change BASE_DIR
        import prepare_data as prep_data
        original_base = prep_data.BASE_DIR
        prep_data.BASE_DIR = tmpdir
        
        try:
            result = prep_data.load_season_league("2023_24", "nba")
            
            assert 'PLAYER_ID' in result.columns
            assert 'PTS' in result.columns
            assert 'POSITION' in result.columns
            assert 'SEASON' in result.columns
            assert 'LEAGUE' in result.columns
            
            assert result.loc[0, 'SEASON'] == "2023_24"
            assert result.loc[0, 'LEAGUE'] == "nba"
        finally:
            prep_data.BASE_DIR = original_base

    def test_merges_on_player_id(self, temp_season_dir):
        """Test that merge happens on PLAYER_ID."""
        tmpdir, season_dir = temp_season_dir
        
        import prepare_data as prep_data
        original_base = prep_data.BASE_DIR
        prep_data.BASE_DIR = tmpdir
        
        try:
            result = prep_data.load_season_league("2023_24", "nba")
            
            # Should only have players in both files
            assert len(result) == 3
            assert all(pid in result['PLAYER_ID'].values for pid in [1, 2, 3])
        finally:
            prep_data.BASE_DIR = original_base


# ============================================================
# INSPECTION TESTS (for debugging/understanding data structure)
# ============================================================

class TestDataFrameInspection:
    """Tests that print out dataframes for manual inspection."""
    
    def test_inspect_sample_stats_df(self, sample_stats_df):
        """Print the sample stats dataframe."""
        print("\n" + "="*80)
        print("SAMPLE STATS DATAFRAME")
        print("="*80)
        print(sample_stats_df)
        print(f"\nShape: {sample_stats_df.shape}")
        print(f"Columns: {sample_stats_df.columns.tolist()}")
        print(f"Data types:\n{sample_stats_df.dtypes}")
    
    def test_inspect_sample_positions_df(self, sample_positions_df):
        """Print the sample positions dataframe."""
        print("\n" + "="*80)
        print("SAMPLE POSITIONS DATAFRAME")
        print("="*80)
        print(sample_positions_df)
        print(f"\nShape: {sample_positions_df.shape}")
        print(f"Columns: {sample_positions_df.columns.tolist()}")
    
    def test_inspect_after_add_features(self, sample_stats_df):
        """Print the dataframe after feature engineering."""
        result = add_features(sample_stats_df)
        
        print("\n" + "="*80)
        print("DATAFRAME AFTER add_features()")
        print("="*80)
        print(result)
        print(f"\nShape: {result.shape}")
        print(f"\nNew columns added:")
        new_cols = set(result.columns) - set(sample_stats_df.columns)
        for col in sorted(new_cols):
            print(f"  {col}: {result[col].dtype}")
        
        print(f"\nSample values for Player A (index 0):")
        print(f"  PTS: {result.loc[0, 'PTS']}")
        print(f"  GP: {result.loc[0, 'GP']}")
        print(f"  PTS_PG: {result.loc[0, 'PTS_PG']:.4f}")
        print(f"  MIN: {result.loc[0, 'MIN']}")
        print(f"  PTS_P36: {result.loc[0, 'PTS_P36']:.4f}")
        print(f"  AST_TO_TOV: {result.loc[0, 'AST_TO_TOV']:.4f}")
        print(f"  FG3_RATE: {result.loc[0, 'FG3_RATE']:.4f}")
    
    def test_inspect_after_add_labels(self, sample_stats_df, sample_positions_df):
        """Print the dataframe after adding position labels."""
        df = sample_stats_df.merge(sample_positions_df, on='PLAYER_ID')
        result = add_labels(df)
        
        print("\n" + "="*80)
        print("DATAFRAME AFTER add_labels()")
        print("="*80)
        print(result[['PLAYER_ID', 'POSITION', 'POSITION_FINE', 'POSITION_COARSE']])
        print(f"\nShape: {result.shape}")
        print(f"\nPosition mapping:")
        for idx, row in result.iterrows():
            print(f"  Player {row['PLAYER_ID']}: {row['POSITION']} → FINE: {row['POSITION_FINE']}, COARSE: {row['POSITION_COARSE']}")
    
    def test_inspect_full_pipeline(self, sample_stats_df, sample_positions_df):
        """Print the complete pipeline output."""
        # Merge stats and positions
        df = sample_stats_df.merge(sample_positions_df, on='PLAYER_ID')
        
        # Apply transformations
        df = add_features(df)
        df = add_labels(df)
        
        print("\n" + "="*80)
        print("FULL PIPELINE OUTPUT")
        print("="*80)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        print(f"\nFirst few rows (all columns):")
        print(df.head())
        
        print(f"\nFeature statistics:")
        print(df[['PTS_PG', 'PTS_P36', 'AST_TO_TOV', 'FG3_RATE']].describe())
        
        print(f"\nPosition distribution:")
        print(f"  Fine-grained: {df['POSITION_FINE'].value_counts().to_dict()}")
        print(f"  Coarse: {df['POSITION_COARSE'].value_counts().to_dict()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s flag shows print statements


