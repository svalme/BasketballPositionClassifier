import time
import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats, CommonPlayerInfo
from nba_api.stats.static import players

API_SLEEP = 0.6

def fetch_league_stats(league: str, season: str) -> pd.DataFrame:
    league_id = "00" if league.lower() == "nba" else "10"

    stats = LeagueDashPlayerStats(
        league_id_nullable=league_id,
        season=season,
        per_mode_detailed="PerGame"
    )

    df = stats.get_data_frames()[0]
    time.sleep(API_SLEEP)
    return df

def get_expected_player_ids(league: str):
    if league.lower() == "nba":
        return {p["id"] for p in players.get_active_players()}
    elif league.lower() == "wnba":
        return {p["id"] for p in players.get_wnba_players()}
    else:
        raise ValueError("league must be 'nba' or 'wnba'")

def save_league_stats(league: str, season: str, out_path: str, skipped_path: str):
    print(f"\nFetching {league.upper()} {season} stats...")

    expected_ids = get_expected_player_ids(league)
    df = fetch_league_stats(league, season)

    found_ids = set(df["PLAYER_ID"])
    skipped_ids = sorted(expected_ids - found_ids)

    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows → {out_path}")
    print(f"Expected players: {len(expected_ids)}")
    print(f"Returned players: {len(found_ids)}")
    print(f"Skipped players: {len(skipped_ids)}")

    # save skipped IDs
    if skipped_ids:
        pd.DataFrame({"PLAYER_ID": skipped_ids}).to_csv(skipped_path, index=False)
        print(f"Skipped IDs saved to {skipped_path}")
    else:
        print("No skipped players")


def fetch_positions(league: str):
    player_ids = get_expected_player_ids(league)

    rows = []
    skipped = []

    for i, pid in enumerate(player_ids, 1):
        try:
            info = CommonPlayerInfo(player_id=pid)
            df = info.get_data_frames()[0]

            if not df.empty and "POSITION" in df.columns:
                pos = df.loc[0, "POSITION"]
                rows.append((pid, pos))
            else:
                skipped.append(pid)

        except Exception:
            skipped.append(pid)

        if i % 25 == 0:
            print(f"{i}/{len(player_ids)} players processed")

        time.sleep(API_SLEEP)

    return pd.DataFrame(rows, columns=["PLAYER_ID", "POSITION"]), skipped

def save_positions_nba(season: str, out_path: str, skipped_path: str):
    df, skipped = fetch_positions("nba")

    df.to_csv(out_path, index=False)
    pd.DataFrame({"PLAYER_ID": skipped}).to_csv(skipped_path, index=False)
    print(f"NBA positions saved.")
    print(f"Valid positions: {len(df)}")
    print(f"Skipped: {len(skipped)}")

from nba_api.stats.endpoints import LeagueDashTeamStats

def get_team_ids(league):
    league_id = "00" if league == "nba" else "10"
    df = LeagueDashTeamStats(league_id_nullable=league_id).get_data_frames()[0]
    return df["TEAM_ID"].unique().tolist()

from nba_api.stats.endpoints import commonteamroster

def fetch_positions_from_rosters(league, season):
    league_id = "00" if league.lower() == "nba" else "10"
    team_ids = get_team_ids(league)
    rows = []
    skipped = []
    for i, team_id in enumerate(team_ids, 1):
        try:
            roster = commonteamroster.CommonTeamRoster(
                league_id_nullable=league_id,
                team_id=team_id,
                season=season
            ).get_data_frames()[0]

            print(f"Fetched roster for team {team_id}, {len(roster)} players.")
            print(roster.head())

            if "POSITION" in roster.columns:
                rows.append(roster[["PLAYER_ID", "POSITION"]])
            else:
                print(f"Team {team_id} has no POSITION data.")
                print(roster.head())
                skipped.append(team_id)

            time.sleep(API_SLEEP)

        except Exception as e:
            print(f"Team {team_id} failed: {e}")
            skipped.append(team_id)

        if i % 5 == 0:
            print(f"{i}/{len(team_ids)} teams processed")

    if not rows:
        print("No positions data found")
        return pd.DataFrame(columns=["PLAYER_ID", "POSITION"]), skipped
    
    positions_df = pd.concat(rows, ignore_index=True).drop_duplicates()
    return positions_df, skipped


def save_positions_wnba(season: str, out_path: str, skipped_path: str):
    df, skipped = fetch_positions_from_rosters("wnba", season)
    df.to_csv(out_path, index=False)
    pd.DataFrame({"TEAM_ID": skipped}).to_csv(skipped_path, index=False)
    print(f"WNBA positions from rosters saved. Total positions: {len(df)}")
    print(f"Skipped teams: {len(skipped)}")


if __name__ == "__main__":

    season = "2024-25"
    season_underscore = season.replace('-', '_')

    # stats retrieved
    file_name_nba_stats = f"data/{season_underscore}/nba_{season_underscore}_stats.csv"
    file_name_wnba_stats = f"data/{season_underscore}/wnba_{season_underscore}_stats.csv"

    # stats skipped
    file_name_wnba_skipped_stats = f"data/{season_underscore}/wnba_{season_underscore}_stats_skipped.csv"
    file_name_nba_skipped_stats = f"data/{season_underscore}/nba_{season_underscore}_stats_skipped.csv"

    save_league_stats("wnba", season, file_name_wnba_stats, file_name_wnba_skipped_stats)
    save_league_stats("nba", season, file_name_nba_stats, file_name_nba_skipped_stats)

    # positions retrieved
    file_name_nba_positions = f"data/{season_underscore}/nba_{season_underscore}_positions.csv"
    file_name_wnba_positions = f"data/{season_underscore}/wnba_{season_underscore}_positions.csv"

    # positions skipped
    file_name_wnba_positions_skipped = f"data/{season_underscore}/wnba_{season_underscore}_positions_skipped.csv"
    file_name_nba_positions_skipped = f"data/{season_underscore}/nba_{season_underscore}_positions_skipped.csv"

    save_positions_nba(season, file_name_nba_positions, file_name_nba_positions_skipped)
    save_positions_wnba(season, file_name_wnba_positions, file_name_wnba_positions_skipped)

