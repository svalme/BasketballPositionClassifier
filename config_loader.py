import yaml

def load_config(env="prod", path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if env not in cfg["environments"]:
        raise ValueError(f"Environment '{env}' not found in config.yaml")
    return cfg["environments"][env]
