# flexseq/temperature/comparison.py

import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from flexseq.data.loader import load_temperature_data
from flexseq.utils.helpers import progress_bar, ProgressCallback
import time

logger = logging.getLogger(__name__)

# --- Constants for Model Comparison ---
# Define the models we want to directly compare.
# Assumes columns like 'neural_network_predicted', 'random_forest_predicted' exist
# in the source all_results.csv files.
MODELS_TO_COMPARE = ['neural_network', 'random_forest']

# ============================================================================
# FUNCTION 1: compare_temperature_predictions (MODIFIED)
# ============================================================================
def compare_temperature_predictions(
    predictions: Dict[Union[int, str], pd.DataFrame],
    config: Dict[str, Any],
    model_names: List[str] # Now expects a list of models
) -> pd.DataFrame:
    """
    Compare predictions across temperatures for multiple models and merge.

    Args:
        predictions: Dictionary mapping temperatures to prediction DataFrames.
        config: Configuration dictionary.
        model_names: List of model names to include predictions for.

    Returns:
        DataFrame with actual values and predictions/errors/uncertainty
        for all specified models across all temperatures.
    """
    if not predictions:
        raise ValueError("No predictions provided for comparison")

    merged_dfs = []
    temperatures = list(predictions.keys())
    logger.info(f"Preparing data for models {model_names} across {len(temperatures)} temps...")

    # Base features common across all residues (take from first available temp df)
    base_feature_cols = ['domain_id', 'resid', 'resname']
    potential_common_features = ['secondary_structure_encoded', 'core_exterior_encoded', 'relative_accessibility', 'normalized_resid']
    first_valid_df = next((df for df in predictions.values() if df is not None), None)
    if first_valid_df is not None:
         for feat in potential_common_features:
              if feat in first_valid_df.columns and feat not in base_feature_cols:
                   base_feature_cols.append(feat)

    for temp in progress_bar(temperatures, desc="Preparing temperature subsets"):
        df = predictions.get(temp) # Use .get for safety
        if df is None or df.empty:
            logger.warning(f"No data found for temperature {temp}. Skipping.")
            continue
        if not all(col in df.columns for col in ['domain_id', 'resid', 'resname']):
             logger.warning(f"Skipping temperature {temp}: Missing essential ID columns.")
             continue

        # Start subset with base features + IDs
        subset_cols = [col for col in base_feature_cols if col in df.columns]
        subset = df[subset_cols].copy()

        # Add actual RMSF for this temp
        target_col = f"rmsf_{temp}" if temp != "average" else "rmsf_average"
        merged_actual_col = f"actual_{temp}"
        if target_col in df.columns:
            subset[merged_actual_col] = df[target_col]
        else:
            logger.warning(f"Actual RMSF column '{target_col}' not found for temperature {temp}")
            subset[merged_actual_col] = np.nan # Add NaN column to ensure consistency

        # Add predictions, errors, uncertainty for EACH requested model
        for model_name in model_names:
            source_pred_col = f"{model_name}_predicted"
            source_error_col = f"{model_name}_error"
            source_abs_error_col = f"{model_name}_abs_error"
            source_uncertainty_col = f"{model_name}_uncertainty"

            # Define merged column names
            merged_pred_col = f"{model_name}_pred_{temp}"
            merged_error_col = f"{model_name}_error_{temp}"
            merged_abs_error_col = f"{model_name}_abs_error_{temp}"
            merged_uncertainty_col = f"{model_name}_unc_{temp}"

            # Add prediction
            if source_pred_col in df.columns:
                subset[merged_pred_col] = df[source_pred_col]
            else:
                # logger.warning(f"'{source_pred_col}' not found for temp {temp}")
                subset[merged_pred_col] = np.nan

            # Add pre-calculated errors if available, otherwise calculate
            if source_abs_error_col in df.columns:
                subset[merged_abs_error_col] = df[source_abs_error_col]
            elif source_pred_col in df.columns and target_col in df.columns:
                 subset[merged_abs_error_col] = (df[source_pred_col] - df[target_col]).abs()
            else: subset[merged_abs_error_col] = np.nan

            if source_error_col in df.columns:
                subset[merged_error_col] = df[source_error_col]
            elif source_pred_col in df.columns and target_col in df.columns:
                 subset[merged_error_col] = df[source_pred_col] - df[target_col]
            else: subset[merged_error_col] = np.nan

            # Add uncertainty
            if source_uncertainty_col in df.columns:
                subset[merged_uncertainty_col] = df[source_uncertainty_col]
            # else: logger.debug(f"'{source_uncertainty_col}' not found for temp {temp}")

        merged_dfs.append(subset)

    if not merged_dfs:
        raise ValueError("No valid dataframes available after preparation for merging.")

    logger.info(f"Merging dataframes for {len(merged_dfs)} temperatures...")
    start_merge_time = time.time()

    # Initialize result with the first DataFrame
    result = merged_dfs[0]
    # Get base feature cols again from the first df actually added
    base_feature_cols = [col for col in base_feature_cols if col in result.columns]

    # Iteratively merge the remaining DataFrames
    for df_to_merge in progress_bar(merged_dfs[1:], desc="Merging temperature data"):
        # Identify columns to keep from the right dataframe (only the temp-specific ones)
        temp_specific_cols = [col for col in df_to_merge.columns if col not in base_feature_cols]
        cols_to_merge = ['domain_id', 'resid', 'resname'] + temp_specific_cols

        try:
            # Merge only the keys and temperature-specific data
            result = pd.merge(
                result, df_to_merge[cols_to_merge],
                on=['domain_id', 'resid', 'resname'],
                how='outer',
                suffixes=('', '_drop_right')
            )
            drop_cols = [col for col in result.columns if col.endswith('_drop_right')]
            if drop_cols: result = result.drop(columns=drop_cols)
        except Exception as e: logger.error(f"Merge step failed: {e}"); raise

    merge_time = time.time() - start_merge_time
    logger.info(f"Finished merging data in {merge_time:.2f}s. Final shape: {result.shape}")

    # --- Add Model Difference Columns ---
    logger.info("Calculating model prediction/error differences...")
    if all(model in model_names for model in MODELS_TO_COMPARE): # Check if both models are present
        m1, m2 = MODELS_TO_COMPARE[0], MODELS_TO_COMPARE[1]
        for temp in temperatures:
            pred1_col = f"{m1}_pred_{temp}"
            pred2_col = f"{m2}_pred_{temp}"
            abs_err1_col = f"{m1}_abs_error_{temp}"
            abs_err2_col = f"{m2}_abs_error_{temp}"

            # Calculate prediction difference if both columns exist
            if pred1_col in result.columns and pred2_col in result.columns:
                result[f"pred_diff_{temp}"] = result[pred1_col] - result[pred2_col]

            # Calculate absolute error difference if both columns exist
            if abs_err1_col in result.columns and abs_err2_col in result.columns:
                result[f"abs_error_diff_{temp}"] = result[abs_err1_col] - result[abs_err2_col]
    else:
        logger.warning(f"Cannot calculate model differences. Required models {MODELS_TO_COMPARE} not found in input model list: {model_names}")


    return result

# ============================================================================
# FUNCTION 2: calculate_temperature_correlations (Unchanged from last version)
# ============================================================================
def calculate_temperature_correlations(
    combined_df: pd.DataFrame,
    temperatures: List[Union[int, str]],
    use_actual: bool = True
) -> pd.DataFrame:
    """
    Calculate correlations between RMSF values at different temperatures.
    (Function content remains the same as the last provided correct version)
    """
    prefix = "actual_" if use_actual else "predicted_"
    valid_temps = [t for t in temperatures if f"{prefix}{t}" in combined_df.columns]
    n_temps = len(valid_temps)
    if n_temps < 2: logger.warning(f"Cannot calculate {prefix} correlations, < 2 valid temp columns."); return pd.DataFrame(index=[str(t) for t in temperatures], columns=[str(t) for t in temperatures])
    corr_matrix = np.full((n_temps, n_temps), np.nan); temp_map = {t: i for i, t in enumerate(valid_temps)}
    for i, temp1 in enumerate(valid_temps):
        col1 = f"{prefix}{temp1}"
        for j, temp2 in enumerate(valid_temps):
            col2 = f"{prefix}{temp2}";
            if i > j: continue
            valid_mask = combined_df[col1].notna() & combined_df[col2].notna()
            if valid_mask.sum() > 1:
                try:
                    x_vals, y_vals = combined_df.loc[valid_mask, col1], combined_df.loc[valid_mask, col2]
                    if x_vals.nunique() > 1 and y_vals.nunique() > 1:
                         pearson_r, _ = stats.pearsonr(x_vals, y_vals); corr_matrix[i, j] = pearson_r;
                         if i != j: corr_matrix[j, i] = pearson_r
                    elif i == j: corr_matrix[i, j] = 1.0
                    else: corr_matrix[i, j] = np.nan; corr_matrix[j, i] = np.nan
                except ValueError: corr_matrix[i, j] = np.nan; corr_matrix[j, i] = np.nan
            elif i == j: corr_matrix[i, j] = 1.0
        corr_matrix[i, i] = 1.0
    corr_df = pd.DataFrame(corr_matrix, index=[str(t) for t in valid_temps], columns=[str(t) for t in valid_temps])
    corr_df = corr_df.reindex(index=[str(t) for t in temperatures], columns=[str(t) for t in temperatures])
    return corr_df

# ============================================================================
# FUNCTION 3: generate_temperature_metrics (MODIFIED for multiple models)
# ============================================================================
def generate_temperature_metrics(
    metrics_input: Dict[Union[int, str], Dict[str, Dict[str, float]]],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generate metrics comparing model performance across temperatures.

    Args:
        metrics_input: Nested dict mapping temperature -> model_name -> metrics.
        config: Configuration dictionary

    Returns:
        DataFrame with metrics for each model and temperature
    """
    metrics_list = config.get("temperature", {}).get("comparison", {}).get("metrics", [])
    if not metrics_list:
         logger.warning("No comparison metrics specified in config, using defaults.")
         metrics_list = ['rmse', 'r2', 'pearson_correlation'] # Default
    results = []
    all_models = set() # Keep track of all models found
    for temp, models_metrics in metrics_input.items():
        for model_name, metrics in models_metrics.items():
            all_models.add(model_name)
            row = {'temperature': temp, 'model': model_name}
            for metric_key in metrics_list:
                row[metric_key] = metrics.get(metric_key, np.nan)
            results.append(row)
    if not results:
        logger.warning("No metrics data generated for temperature comparison.")
        return pd.DataFrame(columns=['temperature', 'model'] + metrics_list)
    result_df = pd.DataFrame(results)
    # Pivot for easier comparison (optional, depends on desired format)
    try:
         # Use pivot_table to handle potential missing model/temp combinations gracefully
         pivot_df = pd.pivot_table(result_df, index='temperature', columns='model', values=metrics_list)
         # Flatten MultiIndex columns if desired: pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
         # Return the pivoted table or the original long format
         # For CSV saving, long format might be more standard. Keep original for now.
         # return pivot_df
         return result_df
    except Exception as e:
         logger.warning(f"Could not pivot metrics table: {e}. Returning long format.")
         return result_df


# ============================================================================
# FUNCTION 4: analyze_temperature_effects (Unchanged from last version)
# ============================================================================
def analyze_temperature_effects(
    combined_df: pd.DataFrame,
    temperatures: List[Union[int, str]],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze how protein flexibility changes with temperature.
    (Function content remains the same as the last provided correct version)
    """
    numeric_temps = [t for t in temperatures if isinstance(t, (int, float)) or (isinstance(t, str) and t.isdigit())]
    if len(numeric_temps) < 2: logger.warning("Not enough numeric temps for trend analysis"); return {}
    numeric_temps = sorted([int(t) if isinstance(t, str) else t for t in numeric_temps]); logger.info(f"Analyzing trends across temps: {numeric_temps}")
    results = {'domain_trends': [], 'residue_outliers': [], 'domain_stats': [], 'aa_responses': [] }; logger.info("Analyzing per-residue temperature trends...")
    start_analysis_time = time.time(); total_residue_groups = len(combined_df[['domain_id', 'resid']].drop_duplicates())
    grouped_residues = progress_bar(combined_df.groupby(['domain_id', 'resid']), desc="Analyzing residue trends", total=total_residue_groups)
    all_slopes_for_stats = []
    for (_, residue_group) in grouped_residues:
        if residue_group.empty: continue
        try: domain_id, resid, resname = residue_group[['domain_id', 'resid', 'resname']].iloc[0]
        except (IndexError, KeyError): logger.warning(f"Could not extract identifiers for a residue group."); continue
        flex_values, temps_used = [], [];
        for temp in numeric_temps:
            actual_col, pred_col = f"actual_{temp}", f"predicted_{temp}" # Note: This uses the *first* model's predicted if actual is missing
            val_to_use = np.nan
            if actual_col in residue_group.columns and pd.notna(residue_group[actual_col].iloc[0]): val_to_use = residue_group[actual_col].iloc[0]
            # elif pred_col in residue_group.columns and pd.notna(residue_group[pred_col].iloc[0]): val_to_use = residue_group[pred_col].iloc[0] # Decide if using predicted for trends is ok
            if pd.notna(val_to_use): flex_values.append(val_to_use); temps_used.append(temp)
        if len(temps_used) < 2: continue
        x, y = np.array(temps_used), np.array(flex_values)
        if np.all(y == y[0]) or np.all(x == x[0]): continue
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            if pd.isna(slope) or pd.isna(r_value): continue
            r_squared = r_value**2
        except ValueError: continue
        results['domain_trends'].append({'domain_id': domain_id, 'resid': resid, 'resname': resname, 'slope': slope, 'intercept': intercept, 'r_squared': r_squared, 'p_value': p_value})
        all_slopes_for_stats.append(slope)
    analysis_time = time.time() - start_analysis_time; logger.info(f"Finished residue trend analysis in {analysis_time:.2f} seconds.")
    if all_slopes_for_stats:
         mean_slope, std_slope = np.nanmean(all_slopes_for_stats), np.nanstd(all_slopes_for_stats); upper_slope_threshold, lower_slope_threshold = mean_slope + 2 * std_slope, 0
         for trend in results['domain_trends']:
              slope = trend['slope']
              if slope < lower_slope_threshold or slope > upper_slope_threshold: results['residue_outliers'].append({'domain_id': trend['domain_id'], 'resid': trend['resid'], 'resname': trend['resname'], 'slope': slope, 'r_squared': trend['r_squared'], 'behavior': 'negative_trend' if slope < 0 else 'high_increase'})
    logger.info("Calculating domain-level trend statistics..."); unique_domains_in_trends = list(set(trend['domain_id'] for trend in results['domain_trends']))
    for domain in progress_bar(unique_domains_in_trends, desc="Analyzing domain trends"):
        domain_trends = [r for r in results['domain_trends'] if r['domain_id'] == domain]; slopes = [r['slope'] for r in domain_trends if pd.notna(r['slope'])]; r_squared = [r['r_squared'] for r in domain_trends if pd.notna(r['r_squared'])]
        if not slopes: continue
        results['domain_stats'].append({'domain_id': domain, 'avg_slope': np.nanmean(slopes), 'std_slope': np.nanstd(slopes), 'avg_r_squared': np.nanmean(r_squared), 'num_residues_analyzed': len(domain_trends), 'outliers_count': len([r for r in results['residue_outliers'] if r['domain_id'] == domain])})
    logger.info("Calculating amino acid-specific trend statistics..."); unique_resnames_in_trends = list(set(trend['resname'] for trend in results['domain_trends']))
    for resname in progress_bar(unique_resnames_in_trends, desc="Analyzing AA trends"):
        resname_trends = [r for r in results['domain_trends'] if r['resname'] == resname]; slopes = [r['slope'] for r in resname_trends if pd.notna(r['slope'])]; r_squared = [r['r_squared'] for r in resname_trends if pd.notna(r['r_squared'])]
        if not slopes: continue
        results['aa_responses'].append({'resname': resname, 'avg_slope': np.nanmean(slopes), 'std_slope': np.nanstd(slopes), 'avg_r_squared': np.nanmean(r_squared), 'num_residues_analyzed': len(resname_trends)})
    logger.info("Finished analyzing temperature effects."); return results

# ============================================================================
# FUNCTION 5: calculate_grouped_error_summary_by_temp (MODIFIED for multiple models)
# ============================================================================
def calculate_grouped_error_summary_by_temp(
    combined_df: pd.DataFrame,
    grouping_col: str,
    temperatures: List[Union[int, str]],
    model_names: List[str], # Now takes list of models
    group_labels: Optional[Dict[Any, str]] = None
) -> pd.DataFrame:
    """
    Calculates error summaries grouped by a specific column, for each temperature, FOR MULTIPLE MODELS.

    Args:
        combined_df: Merged DataFrame with actual and predicted values for multiple models.
        grouping_col: Column name to group by.
        temperatures: List of relevant temperatures.
        model_names: List of model names to include summaries for.
        group_labels: Optional mapping for group labels.

    Returns:
        DataFrame with grouped error summaries per temperature per model.
    """
    if grouping_col not in combined_df.columns:
        logger.warning(f"Grouping column '{grouping_col}' not found. Cannot calculate grouped errors.")
        return pd.DataFrame()

    results = []
    logger.info(f"Calculating error summary grouped by '{grouping_col}' per temperature for models: {model_names}...")

    # Get valid temps common to all requested models (or at least one)
    valid_temps_per_model = {model: [] for model in model_names}
    all_valid_temps = set()
    for temp in temperatures:
        actual_col = f"actual_{temp}"
        if actual_col not in combined_df.columns: continue
        for model in model_names:
            pred_col = f"{model}_pred_{temp}"
            abs_error_col = f"{model}_abs_error_{temp}"
            if pred_col in combined_df.columns and abs_error_col in combined_df.columns:
                 if combined_df[abs_error_col].notna().any():
                      valid_temps_per_model[model].append(temp)
                      all_valid_temps.add(temp)

    if not all_valid_temps:
        logger.warning(f"No valid temperatures with error data found for any model. Skipping grouped summary by {grouping_col}.")
        return pd.DataFrame()

    grouped_data = progress_bar(
        combined_df.groupby(grouping_col, observed=False, dropna=False),
        desc=f"Processing {grouping_col} groups"
    )

    for group_val, group_df in grouped_data:
        group_label = group_labels.get(group_val, str(group_val)) if group_labels else str(group_val)
        count = len(group_df)

        base_info = { 'group_value': group_label, 'count': count }

        # Calculate metrics for each model for each valid temp
        for model in model_names:
            for temp in valid_temps_per_model.get(model, []): # Only loop through valid temps for this model
                abs_error_col = f"{model}_abs_error_{temp}"
                valid_errors = group_df[abs_error_col].dropna()
                if not valid_errors.empty:
                    base_info[f"{model}_mean_abs_error_{temp}"] = valid_errors.mean()
                    base_info[f"{model}_median_abs_error_{temp}"] = valid_errors.median()
                    base_info[f"{model}_std_abs_error_{temp}"] = valid_errors.std()
                else:
                    base_info[f"{model}_mean_abs_error_{temp}"] = np.nan
                    base_info[f"{model}_median_abs_error_{temp}"] = np.nan
                    base_info[f"{model}_std_abs_error_{temp}"] = np.nan

        results.append(base_info)

    if not results: logger.warning(f"No results for grouped error summary by '{grouping_col}'."); return pd.DataFrame()
    summary_df = pd.DataFrame(results).rename(columns={'group_value': grouping_col})
    return summary_df

# ============================================================================
# FUNCTION 6: calculate_feature_binned_errors (MODIFIED for multiple models)
# ============================================================================
def calculate_feature_binned_errors(
    combined_df: pd.DataFrame,
    feature_col: str,
    temperatures: List[Union[int, str]],
    model_names: List[str], # Takes list
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Calculates error summaries binned by a continuous feature, for each temperature, FOR MULTIPLE MODELS.

    Args:
        combined_df: Merged DataFrame.
        feature_col: Continuous feature column name to bin.
        temperatures: List of relevant temperatures.
        model_names: List of model names to include summaries for.
        n_bins: Number of bins.

    Returns:
        DataFrame with binned error summaries per temperature per model.
    """
    if feature_col not in combined_df.columns: logger.warning(f"Feature '{feature_col}' not found."); return pd.DataFrame()
    results = []
    logger.info(f"Calculating error summary binned by '{feature_col}' for models: {model_names}...")

    # Binning
    feature_values = combined_df[feature_col].dropna()
    if feature_values.empty or feature_values.nunique() <= 1: logger.warning(f"Not enough unique values in '{feature_col}' for binning."); return pd.DataFrame()
    bin_col_name = f"{feature_col}_bin_temp"; df_copy = combined_df.copy() # Work on copy for bins
    try:
        bins, bin_edges = pd.cut(feature_values, bins=n_bins, labels=False, retbins=True, include_lowest=True, duplicates='drop')
        df_copy[bin_col_name] = bins.reindex(df_copy.index)
    except Exception as e: logger.error(f"Failed to create bins for '{feature_col}': {e}"); return pd.DataFrame()

    # Valid temps common to all models
    valid_temps_per_model = {model: [] for model in model_names}; all_valid_temps = set()
    for temp in temperatures:
        actual_col = f"actual_{temp}"
        if actual_col not in df_copy.columns: continue
        for model in model_names:
             pred_col = f"{model}_pred_{temp}"
             abs_error_col = f"{model}_abs_error_{temp}" # Need abs error column
             if abs_error_col not in df_copy.columns: # Calculate if missing
                  if pred_col in df_copy.columns: df_copy[abs_error_col] = (df_copy[pred_col] - df_copy[actual_col]).abs()
                  else: df_copy[abs_error_col] = np.nan # Cannot calculate
             if abs_error_col in df_copy.columns and df_copy[abs_error_col].notna().any():
                  valid_temps_per_model[model].append(temp); all_valid_temps.add(temp)
    if not all_valid_temps: logger.warning(f"No valid temps for '{feature_col}' bin summary."); return pd.DataFrame()

    grouped_data = progress_bar(df_copy.groupby(bin_col_name, observed=False, dropna=False), desc=f"Processing {feature_col} bins")
    for bin_idx, group_df in grouped_data:
         if pd.isna(bin_idx): continue
         bin_idx_int = int(bin_idx);
         if bin_idx_int < 0 or bin_idx_int >= len(bin_edges) - 1: continue
         bin_start, bin_end = bin_edges[bin_idx_int], bin_edges[bin_idx_int + 1]
         base_info = {'feature': feature_col, 'bin_start': bin_start, 'bin_end': bin_end, 'bin_center': (bin_start + bin_end) / 2, 'count': len(group_df)}
         for model in model_names:
             for temp in valid_temps_per_model.get(model, []):
                 abs_error_col = f"{model}_abs_error_{temp}"
                 valid_errors = group_df[abs_error_col].dropna()
                 base_info[f"{model}_mean_abs_error_{temp}"] = valid_errors.mean() if not valid_errors.empty else np.nan
         results.append(base_info)

    if not results: logger.warning(f"No results for binned error summary by '{feature_col}'."); return pd.DataFrame()
    summary_df = pd.DataFrame(results)
    return summary_df

# ============================================================================
# FUNCTION 7: calculate_domain_performance_by_temp (MODIFIED for multiple models)
# ============================================================================
def calculate_domain_performance_by_temp(
    combined_df: pd.DataFrame,
    temperatures: List[Union[int, str]],
    model_names: List[str] # Takes list
) -> pd.DataFrame:
    """
    Calculates performance metrics (RMSE, R2, Pearson) per domain, per temperature, FOR MULTIPLE MODELS.

    Args:
        combined_df: Merged DataFrame.
        temperatures: List of relevant temperatures.
        model_names: List of model names to include performance for.

    Returns:
        DataFrame with domain performance metrics per temperature per model.
    """
    results = []
    logger.info(f"Calculating domain performance per temperature for models: {model_names}...")

    # Get valid temps common to all models
    valid_temps_per_model = {model: [] for model in model_names}; all_valid_temps = set()
    for temp in temperatures:
        actual_col = f"actual_{temp}"
        if actual_col not in combined_df.columns: continue
        for model in model_names:
             pred_col = f"{model}_pred_{temp}"
             if pred_col in combined_df.columns and combined_df[pred_col].notna().any():
                  valid_temps_per_model[model].append(temp); all_valid_temps.add(temp)
    if not all_valid_temps: logger.warning("No valid temps for domain performance."); return pd.DataFrame()

    grouped_data = progress_bar(combined_df.groupby('domain_id', observed=False), desc="Processing domains")
    for domain_id, domain_df in grouped_data:
        base_info = {'domain_id': domain_id}
        for temp in sorted(list(all_valid_temps)): # Iterate through all temps that had data for at least one model
            actual_col = f"actual_{temp}"
            # Calculate base metrics for this temp/domain if actual exists
            actual_vals_temp_domain = domain_df[actual_col].dropna()
            base_info[f'num_residues_{temp}'] = len(actual_vals_temp_domain)
            base_info[f'mean_actual_{temp}'] = actual_vals_temp_domain.mean() if not actual_vals_temp_domain.empty else np.nan
            base_info[f'std_actual_{temp}'] = actual_vals_temp_domain.std() if len(actual_vals_temp_domain) > 1 else 0.0

            # Calculate metrics for each model *if* it has valid predictions for this temp
            for model in model_names:
                 if temp in valid_temps_per_model.get(model,[]):
                      pred_col = f"{model}_pred_{temp}"
                      temp_model_df = domain_df[[actual_col, pred_col]].dropna()
                      y_true = temp_model_df[actual_col].values
                      y_pred = temp_model_df[pred_col].values
                      num_eval = len(y_true)

                      rmse, r2, pearson_r = np.nan, np.nan, np.nan
                      if num_eval > 1:
                           try:
                               rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                               if np.var(y_true) > 1e-9 and np.var(y_pred) > 1e-9:
                                    r2 = r2_score(y_true, y_pred)
                                    pearson_r, _ = stats.pearsonr(y_true, y_pred)
                                    if pd.isna(pearson_r): pearson_r = 0.0
                               else:
                                    r2 = 0.0 if np.allclose(y_true, y_pred) else np.nan
                                    pearson_r = 1.0 if np.allclose(y_true, y_pred) else np.nan
                           except ValueError: pass
                      elif num_eval == 1: rmse = np.abs(y_true[0] - y_pred[0])

                      base_info[f"{model}_rmse_{temp}"] = rmse
                      base_info[f"{model}_r2_{temp}"] = r2
                      base_info[f"{model}_pearson_{temp}"] = pearson_r
                 else: # Fill with NaN if model didn't have valid data for this temp
                      base_info[f"{model}_rmse_{temp}"] = np.nan
                      base_info[f"{model}_r2_{temp}"] = np.nan
                      base_info[f"{model}_pearson_{temp}"] = np.nan

        results.append(base_info)

    if not results: logger.warning("No domain performance results generated."); return pd.DataFrame()
    summary_df = pd.DataFrame(results)
    return summary_df


# ============================================================================
# MAIN FUNCTION: prepare_temperature_comparison_data (ENHANCED for new CSVs)
# ============================================================================
def prepare_temperature_comparison_data(
    config: Dict[str, Any],
    model_name: str, # NOTE: This argument is now less relevant, comparison uses MODELS_TO_COMPARE
    output_dir: str
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Prepare data for temperature comparison visualization and analysis.
    Generates multiple CSV files containing processed data suitable for figures.

    Args:
        config: Configuration dictionary.
        model_name: Primary model name (used for fallback/default naming if needed).
        output_dir: Output directory to save comparison data.

    Returns:
        Dictionary mapping data_name -> DataFrame or None if generation failed.
    """
    # Use the hardcoded list of models to compare for this enhanced version
    models_to_analyze = MODELS_TO_COMPARE
    primary_model = model_name if model_name in models_to_analyze else models_to_analyze[0]
    logger.info(f"Starting temperature comparison for models: {models_to_analyze} (Primary: {primary_model})")

    start_prep_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # --- Configurable Parameters ---
    analysis_config = config.get("analysis", {}).get("temperature_comparison", {})
    # Define features for automatic binning (can be customized in config)
    default_features_to_bin = ['relative_accessibility', 'normalized_resid']
    features_to_bin = analysis_config.get("binned_features", default_features_to_bin)
    n_bins_for_features = analysis_config.get("n_bins", 10)

    # --- Load results for each temperature ---
    temperatures = config.get("temperature", {}).get("available", [])
    if not temperatures: logger.error("No available temperatures defined in config."); return {}
    predictions = {}; model_metrics_by_temp_model = {}
    logger.info("Loading results from individual temperature runs...")
    base_output_dir_from_config = config.get("paths", {}).get("output_dir", "./output")

    # Load data and metrics for ALL specified models
    for temp in progress_bar(temperatures, desc="Loading temperature results"):
        temp_output_dir = os.path.join(base_output_dir_from_config, f"outputs_{temp}")
        results_path = os.path.join(temp_output_dir, "all_results.csv")
        if not os.path.exists(results_path): logger.warning(f"Results file not found: {results_path}. Skipping."); continue
        try:
            df_temp = pd.read_csv(results_path)
            # Basic check if essential columns are there
            if not all(c in df_temp for c in ['domain_id', 'resid', 'resname']): continue
            predictions[temp] = df_temp
        except Exception as e: logger.error(f"Failed to load {results_path}: {e}"); continue

        metrics_path = os.path.join(temp_output_dir, "evaluation_results.csv")
        if os.path.exists(metrics_path):
             try:
                metrics_df = pd.read_csv(metrics_path, index_col=0)
                temp_metrics = {}
                for model in models_to_analyze: # Load metrics for all relevant models
                    if model in metrics_df.index:
                        temp_metrics[model] = metrics_df.loc[model].to_dict()
                    else: logger.warning(f"Model '{model}' not found in metrics file: {metrics_path}")
                if temp_metrics: model_metrics_by_temp_model[temp] = temp_metrics # Store {temp: {model: {metrics}}}
             except Exception as e: logger.error(f"Failed to load/parse {metrics_path}: {e}")
        else: logger.warning(f"Metrics file not found: {metrics_path}")

    if not predictions: logger.error("No prediction data loaded. Aborting."); return {}
    logger.info(f"Loaded prediction data for {len(predictions)} temperatures.")

    # --- Create combined predictions dataframe (includes all models) ---
    logger.info("Creating combined predictions dataframe...")
    # Pass the list of models we want to include
    combined_preds = compare_temperature_predictions(predictions, config, models_to_analyze)

    # --- Add explicit error columns if not already present (redundancy check) ---
    valid_temps_present = []
    for temp in temperatures:
        actual_col = f"actual_{temp}"
        if actual_col not in combined_preds.columns: continue # Skip if actual is missing
        temp_had_valid_model_data = False
        for model in models_to_analyze:
            pred_col = f"{model}_pred_{temp}"
            error_col = f"{model}_error_{temp}"
            abs_error_col = f"{model}_abs_error_{temp}"
            if pred_col in combined_preds.columns and combined_preds[pred_col].notna().any():
                 temp_had_valid_model_data = True
                 if error_col not in combined_preds.columns: combined_preds[error_col] = combined_preds[pred_col] - combined_preds[actual_col]
                 if abs_error_col not in combined_preds.columns: combined_preds[abs_error_col] = combined_preds[error_col].abs()
        if temp_had_valid_model_data: valid_temps_present.append(temp)
    logger.info(f"Verified/calculated error columns for temperatures: {valid_temps_present}")

    # --- Save enhanced combined predictions (now includes multiple models and differences) ---
    combined_path = os.path.join(output_dir, "combined_predictions_models_errors_diffs.csv") # More descriptive name
    logger.info(f"Saving combined predictions with multi-model errors/diffs to {combined_path}")
    combined_preds.to_csv(combined_path, index=False)
    generated_data = {'combined_multi_model_predictions': combined_preds} # Store in results

    # --- Calculate and save correlations (Actual vs Actual, Pred vs Pred for primary model) ---
    logger.info("Calculating temperature correlations...")
    actual_corr = calculate_temperature_correlations(combined_preds, valid_temps_present, use_actual=True)
    generated_data['actual_correlations'] = actual_corr # Store DataFrame or None
    if actual_corr is not None and not actual_corr.empty:
        actual_corr_path = os.path.join(output_dir, "actual_correlations.csv"); logger.info(f"Saving actual correlations to {actual_corr_path}"); actual_corr.to_csv(actual_corr_path)

    # Calculate predicted correlations based on the *primary* model for consistency
    predicted_corr = calculate_temperature_correlations(combined_preds, valid_temps_present, use_actual=False) # This now uses predicted_<temp> which corresponds to primary model
    generated_data['predicted_correlations'] = predicted_corr
    if predicted_corr is not None and not predicted_corr.empty:
        predicted_corr_path = os.path.join(output_dir, f"predicted_{primary_model}_correlations.csv"); logger.info(f"Saving {primary_model} predicted correlations to {predicted_corr_path}"); predicted_corr.to_csv(predicted_corr_path)

    # --- Generate and save metrics comparison table (includes multiple models) ---
    logger.info("Generating cross-temperature metrics table...")
    metrics_comparison_df = generate_temperature_metrics(model_metrics_by_temp_model, config)
    generated_data['temperature_metrics_multi_model'] = metrics_comparison_df
    if not metrics_comparison_df.empty:
        metrics_path = os.path.join(output_dir, "temperature_metrics_multi_model.csv"); logger.info(f"Saving multi-model temperature metrics table to {metrics_path}"); metrics_comparison_df.to_csv(metrics_path, index=False)

    # --- Analyze and save temperature effects (based on actual/primary model predicted) ---
    logger.info("Analyzing temperature effects...")
    effects = analyze_temperature_effects(combined_preds, valid_temps_present, config)
    logger.info("Saving temperature effects analysis...")
    for key, data in effects.items():
        if isinstance(data, list) and data:
            df_effects = pd.DataFrame(data); df_path = os.path.join(output_dir, f"{key}.csv"); logger.info(f"Saving {key} to {df_path}"); df_effects.to_csv(df_path, index=False)
            generated_data[key] = df_effects
        # else: logger.warning(f"Skipping saving for effect '{key}'.") # Don't warn for empty lists

    # --- Generate and save grouped/binned error summaries (FOR EACH MODEL) ---
    # 1. By Amino Acid
    aa_error_summary = calculate_grouped_error_summary_by_temp(combined_preds, 'resname', valid_temps_present, models_to_analyze)
    generated_data['aa_error_summary_multi_model'] = aa_error_summary
    if aa_error_summary is not None and not aa_error_summary.empty:
        aa_path = os.path.join(output_dir, f"aa_error_summary_multi_model.csv"); logger.info(f"Saving multi-model amino acid error summary to {aa_path}"); aa_error_summary.to_csv(aa_path, index=False)

    # 2. By Secondary Structure
    if 'secondary_structure_encoded' in combined_preds.columns:
        ss_labels = {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other', -1.0: 'Unknown', np.nan: 'Unknown'} # Handle NaN explicitly
        combined_preds['secondary_structure_encoded'].fillna(-1.0, inplace=True) # Fill NaN before grouping
        ss_error_summary = calculate_grouped_error_summary_by_temp(combined_preds, 'secondary_structure_encoded', valid_temps_present, models_to_analyze, group_labels=ss_labels)
        generated_data['ss_error_summary_multi_model'] = ss_error_summary
        if ss_error_summary is not None and not ss_error_summary.empty:
            ss_path = os.path.join(output_dir, f"ss_error_summary_multi_model.csv"); logger.info(f"Saving multi-model secondary structure error summary to {ss_path}"); ss_error_summary.to_csv(ss_path, index=False)
    else: logger.warning("SS column not found, skipping SS error summary."); generated_data['ss_error_summary_multi_model'] = None

    # 3. By Binned Features (Automatic Detection & Per Model)
    generated_data['feature_binned_errors_multi_model'] = {}
    potential_features = [col for col in combined_preds.columns if combined_preds[col].dtype in [np.float64, np.int64] and not any(str(t) in col for t in temperatures) and col not in ['resid', 'secondary_structure_encoded', 'core_exterior_encoded', 'resname_encoded']] # Basic heuristic
    logger.info(f"Attempting automatic feature binning for: {potential_features}")
    actual_features_to_bin = [f for f in features_to_bin if f in potential_features] # Use configured list if valid
    if not actual_features_to_bin: actual_features_to_bin = [f for f in default_features_to_bin if f in potential_features] # Fallback to default if config list invalid
    logger.info(f"Final features selected for binning: {actual_features_to_bin}")

    for feature in actual_features_to_bin:
         binned_errors = calculate_feature_binned_errors(combined_preds.copy(), feature, valid_temps_present, models_to_analyze, n_bins=n_bins_for_features)
         generated_data['feature_binned_errors_multi_model'][feature] = binned_errors
         if binned_errors is not None and not binned_errors.empty:
             bin_path = os.path.join(output_dir, f"error_by_{feature}_bins_multi_model.csv"); logger.info(f"Saving binned error summary for {feature} to {bin_path}"); binned_errors.to_csv(bin_path, index=False)

    # --- Generate and save domain performance by temp (FOR EACH MODEL) ---
    domain_perf = calculate_domain_performance_by_temp(combined_preds, valid_temps_present, models_to_analyze)
    generated_data['domain_performance_multi_model'] = domain_perf
    if domain_perf is not None and not domain_perf.empty:
        domain_perf_path = os.path.join(output_dir, f"domain_performance_multi_model.csv"); logger.info(f"Saving multi-model domain performance summary to {domain_perf_path}"); domain_perf.to_csv(domain_perf_path, index=False)

    # --- Prepare and save histogram data (Still based on primary model for clarity) ---
    histogram_data = []
    logger.info(f"Generating histogram data (using primary model: {primary_model})...")
    all_actual_rmsf, all_predicted_rmsf = [], []
    for temp in valid_temps_present:
        all_actual_rmsf.extend(combined_preds[f"actual_{temp}"].dropna().tolist())
        all_predicted_rmsf.extend(combined_preds[f"{primary_model}_pred_{temp}"].dropna().tolist())
    if all_actual_rmsf or all_predicted_rmsf:
        q_low = np.percentile(all_actual_rmsf + all_predicted_rmsf, 1) if (all_actual_rmsf + all_predicted_rmsf) else 0
        q_high = np.percentile(all_actual_rmsf + all_predicted_rmsf, 99) if (all_actual_rmsf + all_predicted_rmsf) else 1
        min_val, max_val = max(0, q_low), q_high
        if max_val <= min_val: max_val = min_val + 1e-6
        common_bins = np.linspace(min_val, max_val, 21)
    else: common_bins = np.linspace(0, 1, 21)
    for temp in progress_bar(valid_temps_present, desc="Generating histogram data"):
        actual_col, pred_col = f"actual_{temp}", f"{primary_model}_pred_{temp}"
        actual_values, predicted_values = combined_preds[actual_col].dropna(), combined_preds[pred_col].dropna()
        if not actual_values.empty:
            actual_hist, actual_bin_edges = np.histogram(actual_values, bins=common_bins)
            for i in range(len(actual_hist)): histogram_data.append({'temperature': temp, 'type': 'actual', 'bin_start': actual_bin_edges[i], 'bin_end': actual_bin_edges[i+1], 'count': actual_hist[i]})
        if not predicted_values.empty:
            predicted_hist, predicted_bin_edges = np.histogram(predicted_values, bins=common_bins)
            for i in range(len(predicted_hist)): histogram_data.append({'temperature': temp, 'type': f'predicted_{primary_model}', 'bin_start': predicted_bin_edges[i], 'bin_end': predicted_bin_edges[i+1], 'count': predicted_hist[i]})
    generated_data['histogram_data'] = pd.DataFrame(histogram_data) if histogram_data else None
    if generated_data['histogram_data'] is not None:
        histogram_path = os.path.join(output_dir, "histogram_data.csv"); logger.info(f"Saving histogram data to {histogram_path}"); generated_data['histogram_data'].to_csv(histogram_path, index=False)
    else: logger.warning("No histogram data generated.")

    # --- Final Summary ---
    prep_time = time.time() - start_prep_time
    logger.info(f"Temperature comparison preparation finished in {prep_time:.2f} seconds.")
    logger.info(f"Generated comparison files in: {output_dir}")
    return generated_data