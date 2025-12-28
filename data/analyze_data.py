import pandas as pd


def expand_wnba_positions(pos):
    """Convert WNBA abbreviated positions to full names."""
    mapping = {
        'G': 'Guard',
        'F': 'Forward',
        'C': 'Center',
        'G-F': 'Guard-Forward',
        'F-G': 'Forward-Guard',
        'F-C': 'Forward-Center',
        'C-F': 'Center-Forward',
    }
    return mapping.get(pos, pos)


print("NBA Positions:")
nba_df = pd.read_csv('data/nba_positions.csv')
nba_df = nba_df.dropna(subset=['POSITION']) # remove NaN values
nba_possible_positions = set(nba_df['POSITION'].unique())

print(sorted(nba_possible_positions))
print("NBA Positions:", nba_df['POSITION'].value_counts())

# check for duplicate IDs in NBA
nba_duplicates = nba_df[nba_df['PLAYER_ID'].duplicated(keep=False)]
if len(nba_duplicates) > 0:
    print(f"\nNBA has {len(nba_duplicates)} duplicate IDs:")
    print(nba_duplicates[['PLAYER_ID', 'POSITION']])
else:
    print(f"\nNBA: All {len(nba_df)} player IDs are unique")

print("\nWNBA Positions:")
wnba_df = pd.read_csv('data/wnba_positions.csv')

new_wnba_df = wnba_df.copy()
new_wnba_df['POSITION'] = new_wnba_df['POSITION'].apply(expand_wnba_positions)
new_wnba_possible_positions = set(new_wnba_df['POSITION'].unique())

print(sorted(new_wnba_possible_positions))
print("WNBA Positions:", new_wnba_df['POSITION'].value_counts())

# check for duplicate IDs in WNBA
wnba_duplicates = wnba_df[wnba_df['PLAYER_ID'].duplicated(keep=False)]
if len(wnba_duplicates) > 0:
    print(f"\nWNBA has {len(wnba_duplicates)} duplicate IDs:")
    print(wnba_duplicates[['PLAYER_ID', 'POSITION']])
else:
    print(f"\nWNBA: All {len(wnba_df)} player IDs are unique")

# check for overlap between NBA and WNBA
print("\n" + "="*50)
print("Cross-League Check:")
print("="*50)
nba_ids = set(nba_df['PLAYER_ID'].unique())
wnba_ids = set(wnba_df['PLAYER_ID'].unique())

overlap = nba_ids & wnba_ids
if len(overlap) > 0:
    print(f"\n{len(overlap)} player IDs appear in BOTH NBA and WNBA:")
    print(overlap)
else:
    print(f"\nNo overlap: NBA and WNBA have completely different player IDs")

print(f"\nNBA unique IDs: {len(nba_ids)}")
print(f"WNBA unique IDs: {len(wnba_ids)}")
print(f"Total unique players across both leagues: {len(nba_ids | wnba_ids)}")

