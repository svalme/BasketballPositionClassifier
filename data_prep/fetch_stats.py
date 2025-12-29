import time
import os
import pandas as pd

from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    CommonPlayerInfo,
    LeagueDashTeamStats,
    CommonTeamRoster,
)
from nba_api.stats.static import players

# ============================================================
# CONFIG
# ============================================================

API_SLEEP = 0.6

LEAGUE_IDS = {"nba": "00", "wnba": "10"}

# ============================================================
# UTILS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def get_expected_player_ids(league: str):
    if league == "nba":
        return {p["id"] for p in players.get_active_players()}
    elif league == "wnba":
        return {p["id"] for p in players.get_wnba_players()}
    else:
        raise ValueError("league must be 'nba' or 'wnba'")


# ============================================================
# STATS FETCHING
# ============================================================

def fetch_league_stats(league: str, season: str) -> pd.DataFrame:
    df = LeagueDashPlayerStats(
        league_id_nullable=LEAGUE_IDS[league],
        season=season,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]

    time.sleep(API_SLEEP)
    return df


def save_league_stats(league: str, season: str, out_path: str, skipped_path: str):
    print(f"\nFetching {league.upper()} {season} stats...")

    ensure_dir(out_path)
    expected_ids = get_expected_player_ids(league)
    df = fetch_league_stats(league, season)

    found_ids = set(df["PLAYER_ID"])
    skipped_ids = sorted(expected_ids - found_ids)

    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows: {out_path}")
    print(f"Expected players: {len(expected_ids)}")
    print(f"Returned players: {len(found_ids)}")
    print(f"Skipped players: {len(skipped_ids)}")

    if skipped_ids:
        pd.DataFrame({"PLAYER_ID": skipped_ids}).to_csv(skipped_path, index=False)
        print(f"Skipped IDs saved: {skipped_path}")
    else:
        print("No skipped players")

# ============================================================
# POSITION FETCHING
# ============================================================

def fetch_positions_from_common_info(player_ids):
    rows, skipped = [], []

    for i, pid in enumerate(player_ids, 1):
        try:
            df = CommonPlayerInfo(player_id=pid).get_data_frames()[0]
            if not df.empty and "POSITION" in df.columns:
                rows.append((pid, df.loc[0, "POSITION"]))
            else:
                skipped.append(pid)
        except Exception:
            skipped.append(pid)

        if i % 25 == 0:
            print(f"{i}/{len(player_ids)} players processed")

        time.sleep(API_SLEEP)

    return pd.DataFrame(rows, columns=["PLAYER_ID", "POSITION"]), skipped


def fetch_positions_from_rosters(league: str, season: str):
    team_ids = LeagueDashTeamStats(
        league_id_nullable=LEAGUE_IDS[league]
    ).get_data_frames()[0]["TEAM_ID"].tolist()

    rows, skipped = [], []

    for i, team_id in enumerate(team_ids, 1):
        try:
            df = CommonTeamRoster(
                league_id_nullable=LEAGUE_IDS[league],
                team_id=team_id,
                season=season,
            ).get_data_frames()[0]

            if "POSITION" in df.columns:
                rows.append(df[["PLAYER_ID", "POSITION"]])
            else:
                skipped.append(team_id)

        except Exception:
            skipped.append(team_id)

        if i % 5 == 0:
            print(f"{i}/{len(team_ids)} teams processed")

        time.sleep(API_SLEEP)

    if rows:
        return pd.concat(rows, ignore_index=True).drop_duplicates(), skipped
    return pd.DataFrame(columns=["PLAYER_ID", "POSITION"]), skipped


def save_positions(league: str, season: str, out_path: str, skipped_path: str):
    ensure_dir(out_path)

    if league == "nba":
        df, skipped = fetch_positions_from_common_info(
            get_expected_player_ids("nba")
        )
    else:
        df, skipped = fetch_positions_from_rosters("wnba", season)

    df.to_csv(out_path, index=False)
    pd.DataFrame({"PLAYER_ID": skipped}).to_csv(skipped_path, index=False)

    print(f"{league.upper()} positions saved: {len(df)}")
    print(f"Skipped: {len(skipped)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    season = "2020-21"
    season_tag = season.replace("-", "_")

    base = f"data/{season_tag}"

    save_league_stats(
        "nba",
        season,
        f"{base}/nba_{season_tag}_stats.csv",
        f"{base}/nba_{season_tag}_stats_skipped.csv",
    )

    save_league_stats(
        "wnba",
        season,
        f"{base}/wnba_{season_tag}_stats.csv",
        f"{base}/wnba_{season_tag}_stats_skipped.csv",
    )

    save_positions(
        "nba",
        season,
        f"{base}/nba_{season_tag}_positions.csv",
        f"{base}/nba_{season_tag}_positions_skipped.csv",
    )

    save_positions(
        "wnba",
        season,
        f"{base}/wnba_{season_tag}_positions.csv",
        f"{base}/wnba_{season_tag}_positions_skipped.csv",
    )
