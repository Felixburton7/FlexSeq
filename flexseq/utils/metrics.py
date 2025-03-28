"""
Evaluation metrics for the FlexSeq ML pipeline.

This module provides functions for evaluating model performance
and cross-validation.
"""

import logging
from typing import Dict, List, Any, Union, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    median_absolute_error
)
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Dict[str, Any],
    X: Optional[np.ndarray] = None,
    n_features: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate predictions using multiple metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        config: Configuration dictionary with metrics settings
        X: Optional feature matrix for advanced metrics
        n_features: Optional number of features for adjusted R2
        
    Returns:
        Dictionary of metric names and values
    """
    results = {}
    metrics_config = config["evaluation"]["metrics"]
    
    # Root Mean Squared Error
    if metrics_config.get("rmse", True):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        results["rmse"] = rmse
    
    # Mean Absolute Error
    if metrics_config.get("mae", True):
        mae = mean_absolute_error(y_true, y_pred)
        results["mae"] = mae
    
    # R-squared
    if metrics_config.get("r2", True):
        r2 = r2_score(y_true, y_pred)
        results["r2"] = r2
    
    # Pearson Correlation
    if metrics_config.get("pearson_correlation", True):
        pearson_corr, p_value = pearsonr(y_true, y_pred)
        results["pearson_correlation"] = pearson_corr
        results["pearson_p_value"] = p_value
    
    # Spearman Correlation
    if metrics_config.get("spearman_correlation", True):
        spearman_corr, p_value = spearmanr(y_true, y_pred)
        results["spearman_correlation"] = spearman_corr
        results["spearman_p_value"] = p_value
    
    # Explained Variance Score
    if metrics_config.get("explained_variance", False):
        ev = explained_variance_score(y_true, y_pred)
        results["explained_variance"] = ev
    
    # Max Error
    if metrics_config.get("max_error", False):
        me = max_error(y_true, y_pred)
        results["max_error"] = me
    
    # Median Absolute Error
    if metrics_config.get("median_absolute_error", False):
        medae = median_absolute_error(y_true, y_pred)
        results["median_absolute_error"] = medae
    
    # Adjusted R2
    if metrics_config.get("adjusted_r2", False) and n_features is not None:
        n = len(y_true)
        r2 = results.get("r2", r2_score(y_true, y_pred))
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        results["adjusted_r2"] = adj_r2
    
    # Root Mean Square Absolute Error
    if metrics_config.get("root_mean_square_absolute_error", False):
        rmsae = np.sqrt(np.mean(np.abs(y_true - y_pred)**2))
        results["root_mean_square_absolute_error"] = rmsae
    
    # Q2 (Cross-validated R2)
    if metrics_config.get("q2", False) and X is not None:
        from sklearn.linear_model import LinearRegression
        cv_r2 = cross_val_score(
            LinearRegression(), X, y_true, 
            cv=5, scoring='r2'
        ).mean()
        results["q2"] = cv_r2
    
    # Temperature-specific metrics (for OmniFlex mode)
    if config["mode"]["active"] == "omniflex":
        # Calculate additional metrics for temperature analysis
        temp = config["temperature"]["current"]
        if str(temp).isdigit():
            # Temperature coefficient (how well the model captures temperature effects)
            # This is a placeholder - in a real implementation, this would use
            # actual temperature coefficients from molecular dynamics
            temp_coef = np.corrcoef(y_pred, np.array(y_true) * int(temp)/400)[0, 1]
            results["temperature_coefficient"] = temp_coef
    
    return results

def cross_validate_model(
    model_class: Any,
    model_params: Dict[str, Any],
    data: pd.DataFrame,
    config: Dict[str, Any],
    n_folds: int = 5,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    Perform cross-validation for a model.
    
    Args:
        model_class: Model class to instantiate
        model_params: Parameters for model initialization
        data: DataFrame with features and target
        config: Configuration dictionary
        n_folds: Number of cross-validation folds
        return_predictions: Whether to return predictions
        
    Returns:
        Dictionary with cross-validation results
    """
    from flexseq.data.processor import prepare_data_for_model
    from flexseq.utils.helpers import progress_bar
    
    # Get target column
    target_col = config["dataset"]["target"]
    
    # Initialize metrics storage
    metrics = {
        "rmse": [],
        "mae": [],
        "r2": [],
        "pearson_correlation": []
    }
    
    # Add storage for predictions if requested
    all_predictions = []
    all_true_values = []
    
    # Create cross-validation folds
    stratify_by_domain = config["dataset"]["split"].get("stratify_by_domain", True)
    random_state = config["system"]["random_state"]
    
    if stratify_by_domain:
        # Domain-aware cross-validation
        unique_domains = data["domain_id"].unique()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        fold_domains = []
        for train_idx, test_idx in kf.split(unique_domains):
            # Get train and test domains
            train_domains = unique_domains[train_idx]
            test_domains = unique_domains[test_idx]
            fold_domains.append((train_domains, test_domains))
        
        for i, (train_domains, test_domains) in enumerate(
            progress_bar(fold_domains, desc=f"Cross-validation ({n_folds} folds)")
        ):
            # Split data by domains
            train_data = data[data["domain_id"].isin(train_domains)]
            test_data = data[data["domain_id"].isin(test_domains)]
            
            # Prepare data for model
            X_train, y_train, feature_names = prepare_data_for_model(train_data, config)
            X_test, y_test, _ = prepare_data_for_model(test_data, config)
            
            # Create and train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Evaluate predictions with feature count for adjusted R2
            n_features = X_train.shape[1]
            fold_metrics = evaluate_predictions(
                y_test, y_pred, config, X_test, n_features
            )
            
            # Store results
            for metric, value in fold_metrics.items():
                if metric in metrics:
                    metrics[metric].append(value)
                else:
                    metrics[metric] = [value]
            
            # Store predictions if requested
            if return_predictions:
                all_predictions.extend(y_pred)
                all_true_values.extend(y_test)
                
            # Get uncertainty if available
            if hasattr(model, 'predict_with_std'):
                try:
                    _, y_std = model.predict_with_std(X_test)
                    if "uncertainty_std" not in metrics:
                        metrics["uncertainty_std"] = []
                    metrics["uncertainty_std"].append(np.mean(y_std))
                except Exception as e:
                    logger.warning(f"Could not calculate uncertainty: {e}")
    else:
        # Regular cross-validation
        X, y, feature_names = prepare_data_for_model(data, config)
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        fold_indices = []
        for train_idx, test_idx in kf.split(X):
            fold_indices.append((train_idx, test_idx))
        
        for i, (train_idx, test_idx) in enumerate(
            progress_bar(fold_indices, desc=f"Cross-validation ({n_folds} folds)")
        ):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create and train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Evaluate predictions with feature count for adjusted R2
            n_features = X_train.shape[1]
            fold_metrics = evaluate_predictions(
                y_test, y_pred, config, X_test, n_features
            )
            
            # Store results
            for metric, value in fold_metrics.items():
                if metric in metrics:
                    metrics[metric].append(value)
                else:
                    metrics[metric] = [value]
            
            # Store predictions if requested
            if return_predictions:
                all_predictions.extend(y_pred)
                all_true_values.extend(y_test)
                
            # Get uncertainty if available
            if hasattr(model, 'predict_with_std'):
                try:
                    _, y_std = model.predict_with_std(X_test)
                    if "uncertainty_std" not in metrics:
                        metrics["uncertainty_std"] = []
                    metrics["uncertainty_std"].append(np.mean(y_std))
                except Exception as e:
                    logger.warning(f"Could not calculate uncertainty: {e}")
    
    # Calculate statistics
    results = {}
    
    for metric, values in metrics.items():
        if values:
            results[f"mean_{metric}"] = np.mean(values)
            results[f"std_{metric}"] = np.std(values)
    
    # Add predictions if requested
    if return_predictions:
        results["predictions"] = np.array(all_predictions)
        results["true_values"] = np.array(all_true_values)
    
    return results

def calculate_residue_metrics(
    df: pd.DataFrame,
    target_col: str,
    prediction_cols: List[str],
    include_uncertainty: bool = False,
    uncertainty_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate residue-level metrics for model predictions.
    
    Args:
        df: DataFrame with true values and predictions
        target_col: Column with true target values
        prediction_cols: Columns with model predictions
        include_uncertainty: Whether to include uncertainty metrics
        uncertainty_cols: Columns with prediction uncertainties
        
    Returns:
        DataFrame with residue-level metrics
    """
    results = []
    
    for (domain_id, resid), residue_df in df.groupby(["domain_id", "resid"]):
        residue_metrics = {
            "domain_id": domain_id,
            "resid": resid,
            "resname": residue_df["resname"].iloc[0] if "resname" in residue_df.columns else None
        }
        
        # Add structural features if available
        for feature in ["secondary_structure_encoded", "core_exterior_encoded"]:
            if feature in residue_df.columns:
                residue_metrics[feature] = residue_df[feature].iloc[0]
        
        # Calculate metrics for each model
        true_value = residue_df[target_col].iloc[0]
        residue_metrics["actual"] = true_value
        
        for pred_col in prediction_cols:
            pred_value = residue_df[pred_col].iloc[0]
            residue_metrics[pred_col] = pred_value
            
            # Calculate error
            error = pred_value - true_value
            abs_error = abs(error)
            
            model_name = pred_col.split("_predicted")[0]
            residue_metrics[f"{model_name}_error"] = error
            residue_metrics[f"{model_name}_abs_error"] = abs_error
            
            # Add uncertainty if available
            if include_uncertainty and uncertainty_cols is not None:
                unc_col = next((col for col in uncertainty_cols if model_name in col), None)
                if unc_col and unc_col in residue_df.columns:
                    unc_value = residue_df[unc_col].iloc[0]
                    residue_metrics[f"{model_name}_uncertainty"] = unc_value
                    
                    # Calculate normalized error (error / uncertainty)
                    if unc_value > 0:
                        normalized_error = abs_error / unc_value
                        residue_metrics[f"{model_name}_normalized_error"] = normalized_error
        
        results.append(residue_metrics)
    
    return pd.DataFrame(results)

def calculate_temperature_scaling_factors(
    df: pd.DataFrame,
    temperatures: List[Union[int, str]]
) -> Dict[str, float]:
    """
    Calculate scaling factors between RMSF values at different temperatures.
    
    Args:
        df: DataFrame with RMSF values at multiple temperatures
        temperatures: List of temperatures to analyze
        
    Returns:
        Dictionary mapping temperature pairs to scaling factors
    """
    # Convert string temperatures to int if numeric
    temp_list = []
    for temp in temperatures:
        if isinstance(temp, str) and temp.isdigit():
            temp_list.append(int(temp))
        elif isinstance(temp, int):
            temp_list.append(temp)
    
    # Sort temperatures
    temp_list.sort()
    
    # Calculate scaling factors
    scaling_factors = {}
    
    for i, temp1 in enumerate(temp_list):
        col1 = f"rmsf_{temp1}"
        if col1 not in df.columns:
            continue
            
        for j, temp2 in enumerate(temp_list[i+1:], i+1):
            col2 = f"rmsf_{temp2}"
            if col2 not in df.columns:
                continue
                
            # Calculate average ratio
            valid_mask = (df[col1] > 0) & (df[col2] > 0)
            ratios = df.loc[valid_mask, col2] / df.loc[valid_mask, col1]
            
            # Store average
            key = f"{temp1}_to_{temp2}"
            scaling_factors[key] = ratios.mean()
    
    return scaling_factors

def calculate_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray
) -> Dict[str, float]:
    """
    Calculate metrics for uncertainty quantification.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        y_std: Standard deviation of predictions (uncertainty)
        
    Returns:
        Dictionary of uncertainty metrics
    """
    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)
    
    # Calculate percentage of true values within confidence intervals
    within_1std = np.mean(errors <= y_std)
    within_2std = np.mean(errors <= 2 * y_std)
    within_3std = np.mean(errors <= 3 * y_std)
    
    # Ideal percentages: 68% within 1 std, 95% within 2 std, 99.7% within 3 std
    # Calculate calibration error (how far from ideal)
    cal_error_1std = np.abs(within_1std - 0.68)
    cal_error_2std = np.abs(within_2std - 0.95)
    cal_error_3std = np.abs(within_3std - 0.997)
    
    # Average calibration error
    avg_cal_error = (cal_error_1std + cal_error_2std + cal_error_3std) / 3
    
    # Calculate negative log predictive density (NLPD)
    nlpd = np.mean(0.5 * np.log(2 * np.pi * y_std**2) + 
                   0.5 * ((y_true - y_pred)**2) / (y_std**2))
    
    # Calculate uncertainty-error correlation
    # Ideally, higher uncertainty should correlate with higher error
    unc_err_corr = np.corrcoef(y_std, errors)[0, 1]
    
    return {
        "within_1std": within_1std,
        "within_2std": within_2std,
        "within_3std": within_3std,
        "calibration_error_1std": cal_error_1std,
        "calibration_error_2std": cal_error_2std,
        "calibration_error_3std": cal_error_3std,
        "avg_calibration_error": avg_cal_error,
        "nlpd": nlpd,
        "uncertainty_error_correlation": unc_err_corr
    }

def calculate_domain_performance(
    df: pd.DataFrame,
    target_col: str,
    prediction_cols: List[str]
) -> pd.DataFrame:
    """
    Calculate performance metrics by domain.
    
    Args:
        df: DataFrame with predictions and actual values
        target_col: Target column name
        prediction_cols: Columns with model predictions
        
    Returns:
        DataFrame with domain performance metrics
    """
    results = []
    
    for domain_id, domain_df in df.groupby("domain_id"):
        row = {"domain_id": domain_id, "num_residues": len(domain_df)}
        
        # Add domain properties if available
        if "core_exterior_encoded" in domain_df.columns:
            row["percent_surface"] = domain_df["core_exterior_encoded"].mean() * 100
        
        if "secondary_structure_encoded" in domain_df.columns:
            # Map values to types
            ss_mapping = {0: "helix", 1: "sheet", 2: "loop"}
            ss_counts = domain_df["secondary_structure_encoded"].value_counts(normalize=True) * 100
            for ss_code, ss_type in ss_mapping.items():
                row[f"percent_{ss_type}"] = ss_counts[ss_code] if ss_code in ss_counts else 0
        
        # Calculate metrics for each model
        for pred_col in prediction_cols:
            model_name = pred_col.split("_predicted")[0]
            
            # Calculate metrics
            y_true = domain_df[target_col].values
            y_pred = domain_df[pred_col].values
            
            row[f"{model_name}_rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            row[f"{model_name}_mae"] = mean_absolute_error(y_true, y_pred)
            row[f"{model_name}_r2"] = r2_score(y_true, y_pred)
            
            # Calculate Pearson correlation
            pearson_corr, _ = pearsonr(y_true, y_pred)
            row[f"{model_name}_pearson"] = pearson_corr
        
        results.append(row)
    
    return pd.DataFrame(results)