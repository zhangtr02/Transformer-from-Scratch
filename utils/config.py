import yaml
import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """
    Load configuration from yaml file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config