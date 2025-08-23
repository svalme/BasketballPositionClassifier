import logging

def setup_logging(config, env):
    log_level = getattr(logging, config["logging"]["level"])
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # File handler (always)
    file_handler = logging.FileHandler(config["logging"]["file"])
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler for dev
    if env == "dev":
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)

    return logger
