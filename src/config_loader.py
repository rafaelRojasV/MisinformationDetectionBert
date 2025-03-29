# src/config_loader.py

import yaml
import logging

def load_config(config_path='config_bert_base.yaml'):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"❌ Configuration file {config_path} not found.")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"❌ Error parsing the configuration file: {e}")
        return {}
