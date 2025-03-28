"""
Data processing for the FlexSeq ML pipeline.

This module provides functions for preprocessing protein data,
including feature engineering, window-based feature generation,
and handling missing values.
"""

import os
import logging
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any, Union, Set

import numpy as np
import pandas as pd

from flexseq.data.loader import load_file, load_temperature_data

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean input data by handling missing values with appropriate defaults.
    
    Args:
        df: Input DataFrame with protein data
        
    Returns:
        Cleaned DataFrame with no missing values
    """
    cleaned_df = df.copy()
    
    # Fill missing secondary structure with coil (most common)
    if 'dssp' in cleaned_df.columns:
        cleaned_df['dssp'] = cleaned_df['dssp'].fillna('C')
        
    # Fill missing accessibility with moderate value
    if 'relative_accessibility' in cleaned_df.columns:
        cleaned_df['relative_accessibility'] = pd.to_numeric(
            cleaned_df['relative_accessibility'], errors='coerce'
        ).fillna(0.5)
        
    # Fill missing core_exterior with default "core"
    if 'core_exterior' in cleaned_df.columns:
        cleaned_df['core_exterior'] = cleaned_df['core_exterior'].fillna('core')
        
    # Fill missing phi/psi angles with neutral values
    if 'phi' in cleaned_df.columns:
        cleaned_df['phi'] = pd.to_numeric(cleaned_df['phi'], errors='coerce').fillna(0.0)
    if 'psi' in cleaned_df.columns:
        cleaned_df['psi'] = pd.to_numeric(cleaned_df['psi'], errors='coerce').fillna(0.0)
        
    # Add normalized residue position if missing
    if 'normalized_resid' not in cleaned_df.columns and 'resid' in cleaned_df.columns:
        cleaned_df['normalized_resid'] = cleaned_df.groupby('domain_id')['resid'].transform(
            lambda x: (x - x.min()) / max(x.max() - x.min(), 1)
        )
        
    # Add missing encoded features if needed
    if 'resname' in cleaned_df.columns and 'resname_encoded' not in cleaned_df.columns:
        # Simple ordinal encoding for amino acids
        aa_map = {
            'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5,
            'GLN': 6, 'GLU': 7, 'GLY': 8, 'HSD': 9, 'HSE': 10,
            'HSP': 11, 'HIS': 12, 'ILE': 13, 'LEU': 14, 'LYS': 15,
            'MET': 16, 'PHE': 17, 'SER': 18, 'THR': 19, 'TRP': 20,
            'TYR': 21, 'VAL': 22, 'UNK': 0
        }
        cleaned_df['resname_encoded'] = cleaned_df['resname'].map(
            lambda x: aa_map.get(x, 0) if x else 0
        )
        
    if 'core_exterior' in cleaned_df.columns and 'core_exterior_encoded' not in cleaned_df.columns:
        # Binary encoding for core/exterior
        cleaned_df['core_exterior_encoded'] = cleaned_df['core_exterior'].map(
            lambda x: 0 if x == 'core' else 1
        )
        
    if 'dssp' in cleaned_df.columns and 'secondary_structure_encoded' not in cleaned_df.columns:
        # Encode secondary structure
        ss_map = {
            'H': 0,  # Alpha helix
            'G': 0,  # 3/10 helix (grouped with alpha)
            'I': 0,  # Pi helix (grouped with alpha)
            'E': 1,  # Beta sheet
            'B': 1,  # Beta bridge (grouped with sheet)
            'T': 2,  # Turn
            'S': 2,  # Bend (grouped with turn)
            'C': 2,  # Coil
            '-': 2,  # Unknown (default to coil)
        }
        cleaned_df['secondary_structure_encoded'] = cleaned_df['dssp'].map(
            lambda x: ss_map.get(x, 2) if x else 2
        )
        
    # Add normalized phi/psi angles if needed
    if 'phi' in cleaned_df.columns and 'phi_norm' not in cleaned_df.columns:
        # Normalize angles to [-1, 1] range
        cleaned_df['phi_norm'] = cleaned_df['phi'].map(
            lambda x: (x % 360) / 180 - 1 if pd.notnull(x) else 0
        )
        
    if 'psi' in cleaned_df.columns and 'psi_norm' not in cleaned_df.columns:
        cleaned_df['psi_norm'] = cleaned_df['psi'].map(
            lambda x: (x % 360) / 180 - 1 if pd.notnull(x) else 0
        )
    
    # Handle OmniFlex specific features
    # Fill missing esm_rmsf with mean value
    if 'esm_rmsf' in cleaned_df.columns:
        cleaned_df['esm_rmsf'] = pd.to_numeric(cleaned_df['esm_rmsf'], errors='coerce')
        mean_esm = cleaned_df['esm_rmsf'].mean()
        cleaned_df['esm_rmsf'] = cleaned_df['esm_rmsf'].fillna(mean_esm)
    
    # Fill missing voxel_rmsf with mean value
    if 'voxel_rmsf' in cleaned_df.columns:
        cleaned_df['voxel_rmsf'] = pd.to_numeric(cleaned_df['voxel_rmsf'], errors='coerce')
        mean_voxel = cleaned_df['voxel_rmsf'].mean()
        cleaned_df['voxel_rmsf'] = cleaned_df['voxel_rmsf'].fillna(mean_voxel)
    
    return cleaned_df

def filter_domains(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Filter data based on domain inclusion/exclusion rules.
    
    Args:
        df: Input DataFrame with protein data
        config: Configuration dictionary with domain filtering rules
        
    Returns:
        Filtered DataFrame
    """
    domain_config = config.get("dataset", {}).get("domains", {})
    
    # Start with a copy of the input data
    filtered_df = df.copy()
    
    # Filter by domain inclusion if specified
    include_domains = domain_config.get("include", [])
    if include_domains:
        filtered_df = filtered_df[filtered_df['domain_id'].isin(include_domains)]
        
    # Filter by domain exclusion if specified
    exclude_domains = domain_config.get("exclude", [])
    if exclude_domains:
        filtered_df = filtered_df[~filtered_df['domain_id'].isin(exclude_domains)]
        
    # Filter by protein size if specified
    min_size = domain_config.get("min_protein_size", 0)
    if min_size > 0:
        # Get domain sizes
        domain_sizes = filtered_df.groupby('domain_id').size()
        valid_domains = domain_sizes[domain_sizes >= min_size].index
        filtered_df = filtered_df[filtered_df['domain_id'].isin(valid_domains)]
        
    max_size = domain_config.get("max_protein_size")
    if max_size is not None:
        # Get domain sizes
        domain_sizes = filtered_df.groupby('domain_id').size()
        valid_domains = domain_sizes[domain_sizes <= max_size].index
        filtered_df = filtered_df[filtered_df['domain_id'].isin(valid_domains)]
        
    return filtered_df

def create_window_features(
    df: pd.DataFrame, 
    window_size: int, 
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Create window-based features for each domain, respecting domain boundaries.
    
    Args:
        df: Input DataFrame with protein data
        window_size: Number of residues on each side to include in window
        feature_cols: List of feature column names to use for window features
        
    Returns:
        DataFrame with added window features
    """
    result_df = df.copy()
    
    # Process each domain separately to respect domain boundaries
    for domain_id, domain_df in df.groupby('domain_id'):
        # Sort by residue ID to ensure correct window creation
        domain_df = domain_df.sort_values('resid')
        
        # Create window features for each feature column
        for feature in feature_cols:
            if feature not in domain_df.columns:
                continue
                
            feature_values = domain_df[feature].values
            
            # Create columns for each window position
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue  # Skip current position (already exists)
                    
                col_name = f"{feature}_offset_{offset}"
                
                # Create offset values with appropriate padding
                offset_values = np.full_like(feature_values, np.nan)
                
                if offset < 0:
                    # Negative offset (previous residues)
                    offset_values[-offset:] = feature_values[:offset]
                else:
                    # Positive offset (next residues)
                    offset_values[:-offset] = feature_values[offset:]
                    
                # Update the result dataframe
                result_df.loc[domain_df.index, col_name] = offset_values
    
    # Fill missing window values with defaults
    for col in result_df.columns:
        if '_offset_' in col:
            # Extract base feature name
            base_feature = col.split('_offset_')[0]
            
            # For categorical features use mode, for numeric use median
            if base_feature in ['resname_encoded', 'core_exterior_encoded', 'secondary_structure_encoded']:
                # Use mode (most common value) for categorical
                mode_val = result_df[base_feature].mode()[0]
                result_df[col] = result_df[col].fillna(mode_val)
            else:
                # Use median for numeric features
                median_val = result_df[base_feature].median()
                result_df[col] = result_df[col].fillna(median_val)
    
    return result_df

def process_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Process features with appropriate error handling and fallbacks.
    
    Args:
        df: Input DataFrame with raw protein data
        config: Configuration dictionary
        
    Returns:
        DataFrame with processed features
    """
    try:
        # Start with basic feature processing
        processed_df = clean_data(df)
        
        # Add protein size if not present
        if "protein_size" not in processed_df.columns:
            processed_df["protein_size"] = processed_df.groupby("domain_id")["resid"].transform("count")
        
        # Ensure target column is numeric
        rmsf_col = config["dataset"]["target"]
        if rmsf_col in processed_df.columns:
            processed_df[rmsf_col] = pd.to_numeric(processed_df[rmsf_col], errors='coerce').fillna(0.0)
        else:
            logger.warning(f"Target column '{rmsf_col}' not found in data")
        
        # Filter features based on config
        feature_config = config["dataset"]["features"]
        use_features = feature_config.get("use_features", {})
        
        active_features = []
        for feature, enabled in use_features.items():
            if enabled and feature in processed_df.columns:
                active_features.append(feature)
            elif enabled and feature not in processed_df.columns:
                logger.warning(f"Feature '{feature}' is enabled but not found in data")
        
        # Add window features if enabled
        window_config = feature_config.get("window", {})
        if window_config.get("enabled", False):
            window_size = window_config.get("size", 3)
            processed_df = create_window_features(processed_df, window_size, active_features)
        
        return processed_df
    
    except Exception as e:
        # Log error and return original dataframe if processing fails
        logger.error(f"Feature processing failed: {e}")
        return df

def prepare_data_for_model(
    df: pd.DataFrame, 
    config: Dict[str, Any],
    include_target: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Prepare final data matrices for model training/prediction.
    
    Args:
        df: Processed DataFrame with features
        config: Configuration dictionary
        include_target: Whether to include target variable in output
        
    Returns:
        Tuple of (X, y, feature_names) where y is None if include_target is False
    """
    # Get active features
    feature_config = config["dataset"]["features"]
    use_features = feature_config.get("use_features", {})
    
    active_features = []
    for feature, enabled in use_features.items():
        if enabled and feature in df.columns:
            active_features.append(feature)
    
    # Add window features if enabled
    window_config = feature_config.get("window", {})
    if window_config.get("enabled", False):
        window_cols = [col for col in df.columns if "_offset_" in col]
        active_features.extend(window_cols)
    
    # Ensure all required features exist
    missing_features = [f for f in active_features if f not in df.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        active_features = [f for f in active_features if f in df.columns]
    
    # Prepare feature matrix
    X = df[active_features].values
    
    # Prepare target vector if needed
    y = None
    if include_target:
        target_col = config["dataset"]["target"]
        if target_col in df.columns:
            y = df[target_col].values
        else:
            raise ValueError(f"Target column '{target_col}' not found in data")
    
    return X, y, active_features

@lru_cache(maxsize=4)
def _cached_load_data(data_path: str) -> pd.DataFrame:
    """Load data from file with caching."""
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.tsv'):
        return pd.read_csv(data_path, sep='\t')
    elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
        return pd.read_pickle(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

def load_and_process_data(
    data_path: Optional[str] = None,
    config: Dict[str, Any] = None,
    temperature: Optional[Union[int, str]] = None
) -> pd.DataFrame:
    """
    Load and process data from file or using temperature configuration.
    
    Args:
        data_path: Path to data file (if None, use temperature config)
        config: Configuration dictionary
        temperature: Optional temperature value to override config
        
    Returns:
        Processed DataFrame
        
    Raises:
        FileNotFoundError: If data file not found
        ValueError: If required columns are missing
    """
    if data_path:
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data with caching
        try:
            df = _cached_load_data(data_path)
        except Exception as e:
            raise ValueError(f"Error loading data from {data_path}: {e}")
    else:
        # Use temperature-based loading
        if config is None:
            raise ValueError("Config is required when data_path is not provided")
        
        df = load_temperature_data(config, temperature)
    
    if config:
        # Validate required columns
        required_cols = config["dataset"]["features"]["required"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Try to handle templated column names (e.g., rmsf_{temperature})
            templated_cols = []
            for col in missing_cols:
                if "{temperature}" in col:
                    temp = temperature or config["temperature"]["current"]
                    templated_col = col.replace("{temperature}", str(temp))
                    if templated_col in df.columns:
                        templated_cols.append((col, templated_col))
            
            for orig_col, templated_col in templated_cols:
                missing_cols.remove(orig_col)
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Process features
        processed_df = process_features(df, config)
        
        # Filter domains
        filtered_df = filter_domains(processed_df, config)
        
        return filtered_df
    else:
        # Just clean the data if no config provided
        return clean_data(df)

def split_data(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    Respects domain boundaries when stratify_by_domain is True.
    
    Args:
        df: Processed DataFrame with features
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    split_config = config["dataset"]["split"]
    test_size = split_config.get("test_size", 0.2)
    val_size = split_config.get("validation_size", 0.15)
    random_state = split_config.get("random_state", 42)
    stratify_by_domain = split_config.get("stratify_by_domain", True)
    
    if stratify_by_domain:
        # Get unique domains
        domains = df['domain_id'].unique()
        
        # Split domains into train/test
        test_ratio = test_size
        domains_train, domains_test = train_test_split(
            domains, test_size=test_ratio, random_state=random_state
        )
        
        # Split train domains into train/val
        val_ratio = val_size / (1 - test_ratio)
        domains_train_final, domains_val = train_test_split(
            domains_train, test_size=val_ratio, random_state=random_state
        )
        
        # Create dataframes based on domain splits
        train_df = df[df['domain_id'].isin(domains_train_final)].copy()
        val_df = df[df['domain_id'].isin(domains_val)].copy()
        test_df = df[df['domain_id'].isin(domains_test)].copy()
    else:
        # Regular sample-based splitting
        # First split train/test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        # Then split train/val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=random_state
        )
    
    return train_df, val_df, test_df