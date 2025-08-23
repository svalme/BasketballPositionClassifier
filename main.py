import argparse
from nba_fetcher import (
    fetch_last_season_stat,
    fetch_player_position,
    retry_skipped,
    run_full_fetch,
)
from logger_setup import setup_logging
from config_loader import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Data Fetcher")
    parser.add_argument("--env", choices=["dev", "prod"], default="prod",
                        help="Select environment: dev or prod")
    parser.add_argument("--retry-skipped", choices=["stats", "positions"],
                        help="Retry fetching skipped stats or positions only")
    args = parser.parse_args()

    # Load config
    config = load_config(args.env)

    # Logging setup
    logger = setup_logging(config, args.env)

    # Config values
    OUTPUT_STATS_FILE = config["output_files"]["stats"]
    OUTPUT_POSITIONS_FILE = config["output_files"]["positions"]
    SKIPPED_STATS_FILE = config["skipped_files"]["stats"]
    SKIPPED_POSITIONS_FILE = config["skipped_files"]["positions"]

    API_SLEEP_DELAY = config["api"]["sleep_delay"]
    API_MAX_RETRIES = config["api"]["max_retries"]

    # Run
    if args.retry_skipped == "stats":
        retry_skipped(SKIPPED_STATS_FILE, fetch_last_season_stat, OUTPUT_STATS_FILE,
                      transform_fn=lambda item: item[1])
    elif args.retry_skipped == "positions":
        retry_skipped(SKIPPED_POSITIONS_FILE, fetch_player_position, OUTPUT_POSITIONS_FILE,
                      transform_fn=lambda item: [item[0], item[1]])
    else:
        run_full_fetch(OUTPUT_STATS_FILE, OUTPUT_POSITIONS_FILE,
                       SKIPPED_STATS_FILE, SKIPPED_POSITIONS_FILE)
