"""
Temperature comparison functionality for the FlexSeq ML pipeline.

This module provides functions for comparing protein flexibility
predictions across multiple temperatures.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from flexseq.data.loader import load_temperature_data

logger = logging.getLogger(__name__)

def compare_temperature_predictions(
    predictions: Dict[Union[int, str], pd.DataFrame],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compare predictions across temperatures.
    
    Args:
        predictions: Dictionary mapping temperatures to prediction DataFrames
        config: Configuration dictionary
        
    Returns:
        DataFrame with combined predictions for all temperatures
    """
    if not predictions:
        raise ValueError("No predictions provided for comparison")
    
    # Extract and prepare dataframes for merging
    merged_dfs = []
    
    for temp, df in predictions.items():
        # Create a subset with domain_id, resid, resname, and RMSF predictions
        subset = df[['domain_id', 'resid', 'resname']].copy()
        
        # Get target column and prediction column
        target_col = f"rmsf_{temp}" if temp != "average" else "rmsf_average"
        pred_col = f"{target_col}_predicted"
        
        if target_col in df.columns:
            subset[f"actual_{temp}"] = df[target_col]
        
        if pred_col in df.columns:
            subset[f"predicted_{temp}"] = df[pred_col]
        
        # Add uncertainty column if available
        uncertainty_col = f"{target_col}_uncertainty"
        if uncertainty_col in df.columns:
            subset[f"uncertainty_{temp}"] = df[uncertainty_col]
        
        # Add temperature indicator
        subset['temperature'] = temp
        
        merged_dfs.append(subset)
    
    # Combine all temperature predictions
    if not merged_dfs:
        raise ValueError("No valid prediction dataframes found")
    
    # Find common keys for all dataframes
    result = merged_dfs[0]
    
    for df in merged_dfs[1:]:
        # Use suffixes to avoid column name conflicts
        result = pd.merge(
            result, df, 
            on=['domain_id', 'resid', 'resname'], 
            suffixes=('', '_drop')
        )
        
        # Remove duplicate columns
        drop_cols = [col for col in result.columns if col.endswith('_drop')]
        result = result.drop(columns=drop_cols)
    
    return result

def calculate_temperature_correlations(
    combined_df: pd.DataFrame,
    temperatures: List[Union[int, str]],
    use_actual: bool = True
) -> pd.DataFrame:
    """
    Calculate correlations between RMSF values at different temperatures.
    
    Args:
        combined_df: DataFrame with combined predictions
        temperatures: List of temperatures to compare
        use_actual: Whether to use actual values (True) or predictions (False)
        
    Returns:
        DataFrame with correlation matrix
    """
    # Initialize correlation matrix
    n_temps = len(temperatures)
    corr_matrix = np.zeros((n_temps, n_temps))
    
    # Determine column prefix
    prefix = "actual_" if use_actual else "predicted_"
    
    # Fill correlation matrix
    for i, temp1 in enumerate(temperatures):
        col1 = f"{prefix}{temp1}"
        
        # Check if column exists
        if col1 not in combined_df.columns:
            logger.warning(f"Column {col1} not found, skipping correlations for {temp1}")
            continue
        
        for j, temp2 in enumerate(temperatures):
            col2 = f"{prefix}{temp2}"
            
            # Check if column exists
            if col2 not in combined_df.columns:
                logger.warning(f"Column {col2} not found, skipping correlations for {temp2}")
                continue
            
            # Calculate correlation, handling missing values
            valid_mask = ~(combined_df[col1].isna() | combined_df[col2].isna())
            if valid_mask.sum() > 1:
                pearson_r, _ = stats.pearsonr(
                    combined_df.loc[valid_mask, col1],
                    combined_df.loc[valid_mask, col2]
                )
                corr_matrix[i, j] = pearson_r
            else:
                corr_matrix[i, j] = np.nan
    
    # Create correlation DataFrame
    corr_df = pd.DataFrame(
        corr_matrix,
        index=[str(t) for t in temperatures],
        columns=[str(t) for t in temperatures]
    )
    
    return corr_df

def generate_temperature_metrics(
    predictions: Dict[Union[int, str], Dict[str, Dict[str, float]]],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generate metrics comparing model performance across temperatures.
    
    Args:
        predictions: Nested dict mapping temperature -> model_name -> metrics
        config: Configuration dictionary
        
    Returns:
        DataFrame with metrics for each model and temperature
    """
    # Get metrics of interest
    metrics_list = config["temperature"]["comparison"]["metrics"]
    
    # Create a list to store the results
    results = []
    
    for temp, models in predictions.items():
        for model_name, metrics in models.items():
            # Create a row for each temperature/model combination
            row = {
                'temperature': temp,
                'model': model_name
            }
            
            # Add metrics
            for metric in metrics_list:
                if metric in metrics:
                    row[metric] = metrics[metric]
            
            results.append(row)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df

def analyze_temperature_effects(
    combined_df: pd.DataFrame,
    temperatures: List[Union[int, str]],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze how protein flexibility changes with temperature.
    
    Args:
        combined_df: DataFrame with combined predictions
        temperatures: List of temperatures to analyze
        config: Configuration dictionary
        
    Returns:
        Dictionary with analysis results
    """
    # Only use numeric temperatures for trend analysis
    numeric_temps = [t for t in temperatures if isinstance(t, int) or (isinstance(t, str) and t.isdigit())]
    
    if len(numeric_temps) < 2:
        logger.warning("Not enough numeric temperatures for trend analysis")
        return {}
    
    # Convert string temperatures to int
    numeric_temps = [int(t) if isinstance(t, str) else t for t in numeric_temps]
    
    # Sort temperatures
    numeric_temps.sort()
    
    # Initialize results dictionary
    results = {
        'linear_coefficients': {},
        'r_squared': {},
        'domain_trends': [],
        'residue_outliers': []
    }
    
    # Analyze per-residue trends
    for _, residue_group in combined_df.groupby(['domain_id', 'resid']):
        domain_id = residue_group['domain_id'].iloc[0]
        resid = residue_group['resid'].iloc[0]
        resname = residue_group['resname'].iloc[0]
        
        # Get flexibility values across temperatures
        flex_values = []
        for temp in numeric_temps:
            col = f"actual_{temp}"
            if col in residue_group.columns and not pd.isna(residue_group[col].iloc[0]):
                flex_values.append(residue_group[col].iloc[0])
            else:
                # If missing, use predicted value if available
                pred_col = f"predicted_{temp}"
                if pred_col in residue_group.columns and not pd.isna(residue_group[pred_col].iloc[0]):
                    flex_values.append(residue_group[pred_col].iloc[0])
                else:
                    flex_values.append(np.nan)
        
        # Skip if too many missing values
        if sum(~np.isnan(flex_values)) < 2:
            continue
        
        # Fit linear model
        valid_indices = ~np.isnan(flex_values)
        x = np.array(numeric_temps)[valid_indices]
        y = np.array(flex_values)[valid_indices]
        
        if len(x) < 2:
            continue
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Add result to domain trends
        results['domain_trends'].append({
            'domain_id': domain_id,
            'resid': resid,
            'resname': resname,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value
        })
        
        # Check if this is an outlier (high slope or negative slope)
        if slope < 0 or slope > np.nanmean([r['slope'] for r in results['domain_trends']]) + 2 * np.nanstd([r['slope'] for r in results['domain_trends']]):
            results['residue_outliers'].append({
                'domain_id': domain_id,
                'resid': resid,
                'resname': resname,
                'slope': slope,
                'r_squared': r_value**2,
                'behavior': 'negative_trend' if slope < 0 else 'high_increase'
            })
    
    # Calculate domain-level statistics
    domains = combined_df['domain_id'].unique()
    domain_stats = []
    
    for domain in domains:
        domain_mask = combined_df['domain_id'] == domain
        domain_trends = [r for r in results['domain_trends'] if r['domain_id'] == domain]
        
        if not domain_trends:
            continue
        
        # Get average slope and r_squared for the domain
        slopes = [r['slope'] for r in domain_trends]
        r_squared = [r['r_squared'] for r in domain_trends]
        
        domain_stats.append({
            'domain_id': domain,
            'avg_slope': np.nanmean(slopes),
            'std_slope': np.nanstd(slopes),
            'avg_r_squared': np.nanmean(r_squared),
            'num_residues': len(domain_trends),
            'outliers': len([r for r in results['residue_outliers'] if r['domain_id'] == domain])
        })
    
    # Add domain stats to results
    results['domain_stats'] = domain_stats
    
    # Calculate amino acid-specific responses
    aa_responses = []
    
    for resname in np.unique(combined_df['resname']):
        resname_trends = [r for r in results['domain_trends'] if r['resname'] == resname]
        
        if not resname_trends:
            continue
        
        # Get average slope and r_squared for this amino acid
        slopes = [r['slope'] for r in resname_trends]
        r_squared = [r['r_squared'] for r in resname_trends]
        
        aa_responses.append({
            'resname': resname,
            'avg_slope': np.nanmean(slopes),
            'std_slope': np.nanstd(slopes),
            'avg_r_squared': np.nanmean(r_squared),
            'num_residues': len(resname_trends)
        })
    
    # Add amino acid responses to results
    results['aa_responses'] = aa_responses
    
    return results

def prepare_temperature_comparison_data(
    config: Dict[str, Any],
    model_name: str,
    output_dir: str
) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for temperature comparison visualization.
    
    Args:
        config: Configuration dictionary
        model_name: Model name to use for prediction
        output_dir: Output directory to save comparison data
        
    Returns:
        Dictionary mapping data_name -> DataFrame
    """
    # Get available temperatures
    temperatures = config["temperature"]["available"]
    
    # Load predictions for each temperature
    predictions = {}
    model_metrics = {}
    
    for temp in temperatures:
        # Get temperature-specific output directory
        temp_output_dir = os.path.join(config["paths"]["output_dir"], f"outputs_{temp}")
        
        # Load all results
        results_path = os.path.join(temp_output_dir, "all_results.csv")
        
        if not os.path.exists(results_path):
            logger.warning(f"Results file not found for temperature {temp}: {results_path}")
            continue
        
        # Load predictions
        predictions[temp] = pd.read_csv(results_path)
        
        # Load metrics
        metrics_path = os.path.join(temp_output_dir, "evaluation_results.csv")
        
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path, index_col=0)
            
            if model_name in metrics_df.index:
                model_metrics[temp] = metrics_df.loc[model_name].to_dict()
            else:
                logger.warning(f"Model {model_name} not found in metrics for temperature {temp}")
    
    # Create combined predictions
    combined_preds = compare_temperature_predictions(predictions, config)
    
    # Save combined predictions
    combined_path = os.path.join(output_dir, "combined_predictions.csv")
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    combined_preds.to_csv(combined_path, index=False)
    
    # Calculate correlations
    actual_corr = calculate_temperature_correlations(combined_preds, temperatures, use_actual=True)
    predicted_corr = calculate_temperature_correlations(combined_preds, temperatures, use_actual=False)
    
    # Save correlations
    actual_corr_path = os.path.join(output_dir, "actual_correlations.csv")
    predicted_corr_path = os.path.join(output_dir, "predicted_correlations.csv")
    
    actual_corr.to_csv(actual_corr_path)
    predicted_corr.to_csv(predicted_corr_path)
    
    # Generate metrics comparison
    metrics_df = generate_temperature_metrics({'all_models': model_metrics}, config)
    metrics_path = os.path.join(output_dir, "temperature_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    # Analyze temperature effects
    effects = analyze_temperature_effects(combined_preds, temperatures, config)
    
    # Save temperature effects
    for key, data in effects.items():
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            df_path = os.path.join(output_dir, f"{key}.csv")
            df.to_csv(df_path, index=False)
    
    # Prepare histogram data
    histogram_data = []
    
    for temp in temperatures:
        if temp in predictions:
            df = predictions[temp]
            
            # Get RMSF column
            target_col = f"rmsf_{temp}" if temp != "average" else "rmsf_average"
            pred_col = f"{target_col}_predicted"
            
            if target_col in df.columns and pred_col in df.columns:
                # Calculate histogram
                actual_values = df[target_col].dropna()
                predicted_values = df[pred_col].dropna()
                
                if len(actual_values) > 0 and len(predicted_values) > 0:
                    # Get histogram data
                    actual_hist, actual_bins = np.histogram(actual_values, bins=20)
                    predicted_hist, predicted_bins = np.histogram(predicted_values, bins=20)
                    
                    # Add to results
                    for i in range(len(actual_hist)):
                        histogram_data.append({
                            'temperature': temp,
                            'type': 'actual',
                            'bin_start': actual_bins[i],
                            'bin_end': actual_bins[i+1],
                            'count': actual_hist[i]
                        })
                    
                    for i in range(len(predicted_hist)):
                        histogram_data.append({
                            'temperature': temp,
                            'type': 'predicted',
                            'bin_start': predicted_bins[i],
                            'bin_end': predicted_bins[i+1],
                            'count': predicted_hist[i]
                        })
    
    # Save histogram data
    if histogram_data:
        histogram_df = pd.DataFrame(histogram_data)
        histogram_path = os.path.join(output_dir, "histogram_data.csv")
        histogram_df.to_csv(histogram_path, index=False)
    
    return {
        'combined_predictions': combined_preds,
        'actual_correlations': actual_corr,
        'predicted_correlations': predicted_corr,
        'temperature_metrics': metrics_df,
        'histogram_data': pd.DataFrame(histogram_data) if histogram_data else None
    }