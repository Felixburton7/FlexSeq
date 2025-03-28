"""
Configuration handling for the FlexSeq ML pipeline.

This module provides functions for loading, validating, and managing
configuration settings throughout the pipeline, with special support
for temperature-specific configurations.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml
import re

logger = logging.getLogger(__name__)

def deep_merge(base_dict: Dict, overlay_dict: Dict) -> Dict:
    """
    Recursively merge two dictionaries, with values from overlay_dict taking precedence.
    
    Args:
        base_dict: Base dictionary to merge into
        overlay_dict: Dictionary with values that should override base_dict
        
    Returns:
        Dict containing merged configuration
    """
    result = base_dict.copy()
    
    for key, value in overlay_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result

def get_env_var_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables.
    Environment variables should be prefixed with FLEXSEQ_ and use
    underscore separators for nested keys.
    
    Examples:
        FLEXSEQ_PATHS_DATA_DIR=/path/to/data
        FLEXSEQ_MODELS_RANDOM_FOREST_N_ESTIMATORS=200
        FLEXSEQ_TEMPERATURE_CURRENT=320
        
    Returns:
        Dict containing configuration from environment variables
    """
    config = {}
    
    for key, value in os.environ.items():
        if not key.startswith("FLEXSEQ_"):
            continue
            
        # Remove prefix and convert to lowercase
        key = key[8:].lower()
        
        # Split into parts and create nested dict
        parts = key.split("_")
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set value, converting to appropriate type
        value_part = parts[-1]
        
        # Try to convert to appropriate type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() == "null" or value.lower() == "none":
            value = None
        else:
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Keep as string if not convertible
                pass
                
        current[value_part] = value
        
    return config

def parse_param_overrides(params: List[str]) -> Dict[str, Any]:
    """
    Parse parameter overrides from CLI arguments.
    
    Args:
        params: List of parameter overrides in format "key=value"
        
    Returns:
        Dict containing parameter overrides
    """
    if not params:
        return {}
        
    override_dict = {}
    
    for param in params:
        if "=" not in param:
            logger.warning(f"Ignoring invalid parameter override: {param}")
            continue
            
        key, value = param.split("=", 1)
        
        # Convert value to appropriate type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() == "null" or value.lower() == "none":
            value = None
        else:
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Keep as string if not convertible
                pass
                
        # Split key into parts and create nested dict
        parts = key.split(".")
        current = override_dict
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        current[parts[-1]] = value
        
    return override_dict

def template_config_for_temperature(config: Dict[str, Any], temperature: Union[int, str]) -> Dict[str, Any]:
    """
    Apply temperature-specific templating to configuration.
    
    Replaces {temperature} placeholders in strings with the specified temperature.
    
    Args:
        config: Configuration dictionary
        temperature: Temperature value to use in templating
        
    Returns:
        Dictionary with templated configuration
    """
    # Create deep copy to avoid modifying the original
    import copy
    result = copy.deepcopy(config)
    
    def replace_in_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                replace_in_dict(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        replace_in_dict(item)
                    elif isinstance(item, str):
                        d[key][i] = item.replace("{temperature}", str(temperature))
            elif isinstance(value, str):
                d[key] = value.replace("{temperature}", str(temperature))
                
    replace_in_dict(result)
    
    return result

def load_config(
    config_path: Optional[str] = None,
    param_overrides: Optional[List[str]] = None,
    use_env_vars: bool = True,
    temperature: Optional[Union[int, str]] = None
) -> Dict[str, Any]:
    """
    Load configuration from default and user-provided sources.
    
    Args:
        config_path: Optional path to user config file
        param_overrides: Optional list of parameter overrides
        use_env_vars: Whether to use environment variables
        temperature: Optional temperature value for templating
        
    Returns:
        Dict containing merged configuration
        
    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist
        ValueError: If configuration is invalid
    """
    # Determine default config path
    default_path = os.path.join(os.path.dirname(__file__), "..", "default_config.yaml")
    if not os.path.exists(default_path):
        package_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(package_dir, "..", "default_config.yaml")
        
    # Load default config
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Default config not found at {default_path}")
        
    with open(default_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Overlay user config if provided
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"User config not found at {config_path}")
            
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            config = deep_merge(config, user_config)
            
    # Apply environment variable overrides
    if use_env_vars:
        env_config = get_env_var_config()
        config = deep_merge(config, env_config)
        
    # Apply CLI parameter overrides
    if param_overrides:
        override_config = parse_param_overrides(param_overrides)
        config = deep_merge(config, override_config)
    
    # Apply temperature override if provided
    if temperature is not None:
        config["temperature"]["current"] = temperature
    
    # Get current temperature from config
    current_temp = config["temperature"]["current"]
    
    # Apply temperature templating
    config = template_config_for_temperature(config, current_temp)
    
    # Handle OmniFlex mode settings
    if config["mode"]["active"].lower() == "omniflex":
        # Enable ESM and voxel features if in OmniFlex mode
        if config["mode"]["omniflex"]["use_esm"]:
            config["dataset"]["features"]["use_features"]["esm_rmsf"] = True
        
        if config["mode"]["omniflex"]["use_voxel"]:
            config["dataset"]["features"]["use_features"]["voxel_rmsf"] = True
    
    # Validate config (basic validation)
    validate_config(config)
    
    # Set system-wide logging level
    log_level = config.get("system", {}).get("log_level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.getLogger().setLevel(numeric_level)
    
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """
    Perform basic validation of configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ["paths", "dataset", "models", "evaluation", "system", "temperature", "mode"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
            
    # Validate dataset section
    if "target" not in config["dataset"]:
        raise ValueError("Missing required dataset.target configuration")
    
    # Validate temperature section
    current_temp = config["temperature"]["current"]
    available_temps = config["temperature"]["available"]
    
    if str(current_temp) not in [str(t) for t in available_temps]:
        raise ValueError(f"Current temperature {current_temp} is not in the list of available temperatures")
    
    # Check that at least one model is enabled
    any_model_enabled = False
    for model_name, model_config in config.get("models", {}).items():
        if model_name != "common" and isinstance(model_config, dict) and model_config.get("enabled", False):
            any_model_enabled = True
            break
            
    if not any_model_enabled:
        logger.warning("No models are enabled in configuration")
        
    # Additional validation could be added as needed
            
def get_enabled_models(config: Dict[str, Any]) -> List[str]:
    """
    Get list of enabled model names from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of enabled model names
    """
    enabled_models = []
    
    for model_name, model_config in config.get("models", {}).items():
        if model_name != "common" and isinstance(model_config, dict) and model_config.get("enabled", False):
            enabled_models.append(model_name)
            
    return enabled_models

def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model, with common settings applied.
    
    Args:
        config: Full configuration dictionary
        model_name: Name of the model
        
    Returns:
        Model-specific configuration with common settings merged in
        
    Raises:
        ValueError: If model_name is not found in configuration
    """
    models_config = config.get("models", {})
    
    if model_name not in models_config:
        raise ValueError(f"Model '{model_name}' not found in configuration")
        
    model_config = models_config[model_name]
    common_config = models_config.get("common", {})
    
    # Merge common config with model-specific config
    merged_config = deep_merge(common_config, model_config)
    
    return merged_config

def get_available_temperatures(config: Dict[str, Any]) -> List[Union[int, str]]:
    """
    Get list of available temperatures from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of available temperatures
    """
    return config["temperature"]["available"]

def get_output_dir_for_temperature(config: Dict[str, Any], temperature: Union[int, str]) -> str:
    """
    Get output directory path for a specific temperature.
    
    Args:
        config: Configuration dictionary
        temperature: Temperature value
        
    Returns:
        Path to output directory for the specified temperature
    """
    base_output_dir = config["paths"]["output_dir"]
    return os.path.join(base_output_dir, f"outputs_{temperature}")

def get_models_dir_for_temperature(config: Dict[str, Any], temperature: Union[int, str]) -> str:
    """
    Get models directory path for a specific temperature.
    
    Args:
        config: Configuration dictionary
        temperature: Temperature value
        
    Returns:
        Path to models directory for the specified temperature
    """
    base_models_dir = config["paths"]["models_dir"]
    return os.path.join(base_models_dir, f"models_{temperature}")

def get_comparison_output_dir(config: Dict[str, Any]) -> str:
    """
    Get output directory path for temperature comparisons.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to output directory for temperature comparisons
    """
    base_output_dir = config["paths"]["output_dir"]
    return os.path.join(base_output_dir, "outputs_comparison")