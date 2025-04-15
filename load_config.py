import os

def load_config(config_file="config.txt"):
    """Load API keys and settings from a text file into environment variables."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' not found!")

    with open(config_file, "r") as file:
        for line in file:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value  # Store in environment variables

    print("✅ Configuration loaded successfully!")

# Load config on module import
load_config()
