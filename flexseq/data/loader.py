"""
Data loading utilities for the FlexSeq ML pipeline.

This module provides functions for loading protein data from various formats,
with special support for temperature-specific files.
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from functools import lru_cache

import pandas as pd
import numpy as np
import glob


logger = logging.getLogger(__name__)

def list_data_files(data_dir: str, file_pattern: str) -> List[str]:
    """
    List data files matching a pattern in a directory.
    
    Args:
        data_dir: Directory to search
        file_pattern: File pattern to match
        
    Returns:
        List of file paths
    """
    
    # Get absolute path
    data_dir = os.path.abspath(data_dir)
    
    # Find matching files
    pattern = os.path.join(data_dir, file_pattern)
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        logger.warning(f"No files found matching pattern {pattern}")
    
    return matching_files

def get_temperature_files(data_dir: str, file_pattern: str = "temperature_*.csv") -> Dict[Union[int, str], str]:
    """
    Get a dictionary mapping temperature values to file paths.
    
    Args:
        data_dir: Directory to search
        file_pattern: File pattern to match
        
    Returns:
        Dictionary mapping temperature values to file paths
    """
    # Get all matching files
    files = list_data_files(data_dir, file_pattern)
    
    # Extract temperatures from filenames
    temperature_files = {}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Try to extract temperature from filename
        # Pattern: temperature_{temp}_train.csv
        match = re.match(r"temperature_(\d+|average)_.*\.csv", filename)
        
        if match:
            temp_str = match.group(1)
            
            # Convert to int if numeric, keep as string for "average"
            temperature = int(temp_str) if temp_str.isdigit() else temp_str
            
            temperature_files[temperature] = file_path
    
    if not temperature_files:
        logger.warning(f"No temperature files found in {data_dir}")
    
    return temperature_files

def detect_file_format(file_path: str) -> str:
    """
    Detect file format based on extension and content.
    
    Args:
        file_path: Path to data file
        
    Returns:
        Format string ('csv', 'tsv', 'pickle', etc.)
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.csv':
        return 'csv'
    elif ext == '.tsv':
        return 'tsv'
    elif ext in ['.pkl', '.pickle']:
        return 'pickle'
    elif ext == '.json':
        return 'json'
    elif ext == '.parquet':
        return 'parquet'
    elif ext == '.h5':
        return 'hdf5'
    else:
        # Try to detect CSV/TSV by reading first line
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    return 'tsv'
                elif ',' in first_line:
                    return 'csv'
        except:
            pass
        
        # Default to CSV if can't determine
        logger.warning(f"Could not determine format for {file_path}, defaulting to CSV")
        return 'csv'

@lru_cache(maxsize=16)
def load_file(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a file with format auto-detection.
    
    Args:
        file_path: Path to data file
        **kwargs: Additional arguments to pass to pandas
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported or loading fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Detect format
    file_format = detect_file_format(file_path)
    
    try:
        # Load based on format
        if file_format == 'csv':
            return pd.read_csv(file_path, **kwargs)
        elif file_format == 'tsv':
            return pd.read_csv(file_path, sep='\t', **kwargs)
        elif file_format == 'pickle':
            return pd.read_pickle(file_path, **kwargs)
        elif file_format == 'json':
            return pd.read_json(file_path, **kwargs)
        elif file_format == 'parquet':
            return pd.read_parquet(file_path, **kwargs)
        elif file_format == 'hdf5':
            return pd.read_hdf(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise ValueError(f"Failed to load file {file_path}: {e}")

def merge_data_files(file_paths: List[str], **kwargs) -> pd.DataFrame:
    """
    Merge multiple data files into a single DataFrame.
    
    Args:
        file_paths: List of paths to data files
        **kwargs: Additional arguments to pass to pandas
        
    Returns:
        Merged DataFrame
    """
    if not file_paths:
        raise ValueError("No files provided for merging")
    
    # Load and concatenate files
    dfs = []
    
    for file_path in file_paths:
        try:
            df = load_file(file_path, **kwargs)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Skipping file {file_path} due to error: {e}")
    
    if not dfs:
        raise ValueError("No data files could be loaded")
    
    return pd.concat(dfs, ignore_index=True)

def validate_data_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all required columns are present, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        return False
    
    return True

def load_temperature_data(
    config: Dict[str, Any],
    temperature: Optional[Union[int, str]] = None
) -> pd.DataFrame:
    """
    Load data for a specific temperature.
    
    Args:
        config: Configuration dictionary
        temperature: Optional temperature to load (if None, use config["temperature"]["current"])
        
    Returns:
        DataFrame for the specified temperature
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If temperature is not in available temperatures
    """
    # Get temperature from config if not provided
    if temperature is None:
        temperature = config["temperature"]["current"]
    
    # Validate temperature
    available_temps = config["temperature"]["available"]
    if str(temperature) not in [str(t) for t in available_temps]:
        raise ValueError(f"Temperature {temperature} is not in the list of available temperatures: {available_temps}")
    
    # Get data directory and file pattern
    data_dir = config["paths"]["data_dir"]
    file_pattern = config["dataset"]["file_pattern"]
    
    # Replace {temperature} placeholder in file pattern
    file_pattern = file_pattern.replace("{temperature}", str(temperature))
    
    # Find matching file
    matching_files = list_data_files(data_dir, file_pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"No data file found for temperature {temperature} with pattern {file_pattern}")
    
    # Use the first matching file
    file_path = matching_files[0]
    
    # Load data
    df = load_file(file_path)
    
    return df

def load_all_temperature_data(config: Dict[str, Any]) -> Dict[Union[int, str], pd.DataFrame]:
    """
    Load data for all available temperatures.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping temperature values to DataFrames
    """
    # Get available temperatures
    available_temps = config["temperature"]["available"]
    
    # Load data for each temperature
    temperature_data = {}
    
    for temp in available_temps:
        try:
            df = load_temperature_data(config, temp)
            temperature_data[temp] = df
        except Exception as e:
            logger.warning(f"Error loading data for temperature {temp}: {e}")
    
    if not temperature_data:
        raise ValueError("No temperature data could be loaded")
    
    return temperature_data

def summarize_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for a dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of summary statistics
    """
    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "memory_usage": None,
        "domains": None,
        "residues_per_domain": None,
        "column_types": {},
        "missing_values": {},
    }
    
    # Memory usage
    try:
        memory_bytes = df.memory_usage(deep=True).sum()
        
        if memory_bytes < 1024:
            summary["memory_usage"] = f"{memory_bytes} bytes"
        elif memory_bytes < 1024**2:
            summary["memory_usage"] = f"{memory_bytes / 1024:.2f} KB"
        elif memory_bytes < 1024**3:
            summary["memory_usage"] = f"{memory_bytes / (1024**2):.2f} MB"
        else:
            summary["memory_usage"] = f"{memory_bytes / (1024**3):.2f} GB"
    except:
        pass
    
    # Domain statistics if domain_id is present
    if "domain_id" in df.columns:
        domains = df["domain_id"].unique()
        summary["domains"] = {
            "count": len(domains),
            "examples": list(domains[:5])
        }
        
        # Residues per domain
        residue_counts = df.groupby("domain_id").size()
        summary["residues_per_domain"] = {
            "min": residue_counts.min(),
            "max": residue_counts.max(),
            "mean": residue_counts.mean(),
            "median": residue_counts.median()
        }
    
    # Column types and missing values
    for col in df.columns:
        summary["column_types"][col] = str(df[col].dtype)
        missing = df[col].isna().sum()
        if missing > 0:
            summary["missing_values"][col] = {
                "count": missing,
                "percentage": (missing / len(df)) * 100
            }
    
    # Check for temperature-specific RMSF columns
    rmsf_columns = [col for col in df.columns if col.startswith("rmsf_")]
    if rmsf_columns:
        summary["rmsf_columns"] = rmsf_columns
    
    # Check for OmniFlex specific columns
    omniflex_columns = ["esm_rmsf", "voxel_rmsf"]
    found_omniflex_columns = [col for col in omniflex_columns if col in df.columns]
    if found_omniflex_columns:
        summary["omniflex_columns"] = found_omniflex_columns
    
    return summary

def log_data_summary(summary: Dict[str, Any]) -> None:
    """
    Log a summary of dataset statistics.
    
    Args:
        summary: Dictionary of summary statistics
    """
    logger.info("=== Dataset Summary ===")
    logger.info(f"Rows: {summary['num_rows']}, Columns: {summary['num_columns']}")
    
    if "memory_usage" in summary and summary["memory_usage"]:
        logger.info(f"Memory usage: {summary['memory_usage']}")
    
    if "domains" in summary and summary["domains"]:
        logger.info(f"Domains: {summary['domains']['count']} unique domains")
        logger.info(f"Examples: {', '.join(summary['domains']['examples'])}")
    
    if "residues_per_domain" in summary and summary["residues_per_domain"]:
        stats = summary["residues_per_domain"]
        logger.info(f"Residues per domain: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}")
    
    if "missing_values" in summary and summary["missing_values"]:
        missing = summary["missing_values"]
        if missing:
            logger.info("Columns with missing values:")
            for col, stats in missing.items():
                logger.info(f"  {col}: {stats['count']} missing ({stats['percentage']:.1f}%)")
        else:
            logger.info("No missing values detected")
    
    if "rmsf_columns" in summary:
        logger.info(f"RMSF columns: {', '.join(summary['rmsf_columns'])}")
    
    if "omniflex_columns" in summary:
        logger.info(f"OmniFlex columns: {', '.join(summary['omniflex_columns'])}")
    
    logger.info("========================")