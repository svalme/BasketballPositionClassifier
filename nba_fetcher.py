import time
import logging
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats, commonplayerinfo
import csv

API_SLEEP_DELAY = 0.5


# -----------------------------
# CSV Writer
# -----------------------------
def write_csv(filename, rows, mode="w", header=None):
    if not rows:
        logging.info(f"Nothing to write for {filename}, skipping file creation.")
        return
    with open(filename, mode, newline="") as file:
        writer = csv.writer(file)
        if header and mode == "w":  # only write header on fresh file
            writer.writerow(header)
        writer.writerows(rows)
    logging.info(f"Saved {len(rows)} rows to {filename}")

# -----------------------------
# Fetch Functions
# -----------------------------
def get_active_player_ids():
    active_players = players.get_active_players()
    return [p['id'] for p in active_players]

def fetch_last_season_stat(player_id):
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        df = career.get_data_frames()[0]
        if not df.empty:
            return df.values.tolist()[-1], df.columns.tolist()
    except Exception as e:
        logging.warning(f"Error fetching stats for player {player_id}: {e}")
    return None, None

def fetch_player_position(player_id):
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df = player_info.get_data_frames()[0]
        if not df.empty and "POSITION" in df.columns:
            return df["POSITION"].iloc[0], ["PlayerID", "Position"]
    except Exception as e:
        logging.warning(f"Error fetching position for player {player_id}: {e}")
    return None, None

# -----------------------------
# Retry logic with progress
# -----------------------------
API_MAX_RETRIES = 3
def fetch_with_retry(ids, fetch_fn):
    results = {}
    skipped = ids
    total = len(ids)
    header = None

    for attempt in range(API_MAX_RETRIES):
        logging.info(f"Retry attempt {attempt+1}/{API_MAX_RETRIES}, {len(skipped)} players left.")
        new_skipped = []
        for idx, player_id in enumerate(skipped, start=1):
            try:
                data, cols = fetch_fn(player_id)
                if data:
                    results[player_id] = data
                    if header is None and cols is not None:
                        header = cols
                else:
                    new_skipped.append(player_id)
            except Exception as e:
                logging.warning(f"Error fetching player {player_id}: {e}")
                new_skipped.append(player_id)

            time.sleep(API_SLEEP_DELAY)

            # Live progress counter (console)
            print(f"\rFetched {len(results)}/{total} players, {len(new_skipped)} skipped...", end="")

        skipped = new_skipped
        print()  # newline after each retry attempt

        if not skipped:
            break  # all done early

    if skipped:
        logging.warning(f"Still skipped {len(skipped)} players after {API_MAX_RETRIES} retries.")
    return results, skipped, header

# -----------------------------
# Skipped players utilities
# -----------------------------
def read_skipped(filename):
    try:
        with open(filename, "r") as file:
            return [int(line.strip()) for line in file if line.strip()]
    except FileNotFoundError:
        logging.info(f"No skipped file found: {filename}")
        return []

def retry_skipped(skipped_file, fetch_fn, output_file, transform_fn=lambda x: x):
    skipped_ids = read_skipped(skipped_file)
    if not skipped_ids:
        logging.info(f"No skipped players found in {skipped_file}.")
        open(skipped_file, "w").close()
        return

    results, still_skipped, header = fetch_with_retry(skipped_ids, fetch_fn)
    rows = [transform_fn(item) for item in results.items()]

    # Append rows without writing header again
    write_csv(output_file, rows, mode="a")

    if still_skipped:
        write_csv(skipped_file, [[pid] for pid in still_skipped], mode="w", header=["PlayerID"])
    else:
        open(skipped_file, "w").close()
        logging.info(f"All previously skipped players in {skipped_file} were fetched successfully.")

# -----------------------------
# Generic full fetch
# -----------------------------
def run_fetch(player_ids, fetch_fn, output_file, skipped_file, transform_fn=lambda x: x):
    results, skipped, header = fetch_with_retry(player_ids, fetch_fn)
    rows = [transform_fn(item) for item in results.items()] if results else []
    if header:
        write_csv(output_file, rows, mode="w", header=header)

    if skipped:
        write_csv(skipped_file, [[pid] for pid in skipped], mode="w", header=["PlayerID"])
    else:
        open(skipped_file, "w").close()
        logging.info(f"No skipped players for {fetch_fn.__name__}.")

def run_full_fetch(output_stats_file, output_positions_file, skipped_stats_file, skipped_positions_file):
    player_ids = get_active_player_ids()
    #print(player_ids)
    run_fetch(player_ids, fetch_last_season_stat, output_stats_file, skipped_stats_file,
              transform_fn=lambda item: item[1])
    #print(output_stats_file)
    #run_fetch(player_ids, fetch_player_position, output_positions_file, skipped_positions_file,
              #transform_fn=lambda item: [item[0], item[1]])
