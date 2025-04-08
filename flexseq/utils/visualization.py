"""
Visualization functions for the FlexSeq ML pipeline.

This module provides placeholder functions for generating visualization data
to be used by external visualization tools. The functions primarily save data
in CSV format that can be visualized separately.
"""


from scipy import stats

from flexseq.data.loader import load_temperature_data
# Add these lines:
from flexseq.utils.helpers import progress_bar, ProgressCallback
import time # Optional, but useful for timing sections


import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from flexseq.utils.helpers import (
    get_temperature_color,
    make_model_color_map,
    ensure_dir, 
    progress_bar, 
    ProgressCallback
)

logger = logging.getLogger(__name__)

def save_plot(plt, output_path: str, dpi: int = 300) -> None:
    """
    Save a matplotlib plot to disk.
    
    Args:
        plt: Matplotlib pyplot instance
        output_path: Path to save the plot
        dpi: Resolution in dots per inch
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_temperature_comparison(
    temperatures: List[Union[int, str]],
    metrics: pd.DataFrame,
    output_path: str
) -> None:
    """
    Generate a plot comparing metrics across temperatures.
    
    Args:
        temperatures: List of temperature values
        metrics: DataFrame with metrics for each temperature
        output_path: Path to save the plot
    """
    # Prepare the data for plotting
    metrics_data = []
    for temp in temperatures:
        if str(temp) in metrics.index.astype(str):
            row = metrics.loc[str(temp)]
            metrics_data.append({
                'temperature': temp,
                'rmse': row.get('rmse', np.nan),
                'r2': row.get('r2', np.nan),
                'pearson_correlation': row.get('pearson_correlation', np.nan)
            })
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the data as CSV for external visualization
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(output_path, index=False)
    
    logger.info(f"Temperature comparison data saved to {output_path}")

def plot_amino_acid_performance(
    data: pd.DataFrame,
    temperature: Union[int, str],
    output_path: str
) -> None:
    """
    Generate amino acid-specific performance data.
    
    Args:
        data: DataFrame with predictions and errors by amino acid
        temperature: Temperature value for the data
        output_path: Path to save the data
    """
    # Group data by amino acid
    if 'resname' in data.columns:
        error_cols = [col for col in data.columns if col.endswith('_abs_error')]
        aa_performance = []
        
        for resname, group in data.groupby('resname'):
            row = {'resname': resname, 'count': len(group)}
            
            for error_col in error_cols:
                model_name = error_col.split('_abs_error')[0]
                row[f"{model_name}_mean_error"] = group[error_col].mean()
                row[f"{model_name}_median_error"] = group[error_col].median()
                row[f"{model_name}_std_error"] = group[error_col].std()
            
            aa_performance.append(row)
        
        # Create DataFrame and save to CSV
        aa_df = pd.DataFrame(aa_performance)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        aa_df.to_csv(output_path, index=False)
        logger.info(f"Amino acid performance data saved to {output_path}")

def plot_feature_importance(
    importances: Dict[str, float],
    feature_names: List[str],
    output_path: str
) -> None:
    """
    Generate feature importance visualization data.
    
    Args:
        importances: Dictionary mapping features to importance values
        feature_names: List of feature names
        output_path: Path to save the data
    """
    # Create DataFrame from importance dictionary
    importance_data = []
    
    for feature, importance in importances.items():
        importance_data.append({
            'feature': feature,
            'importance': importance
        })
    
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    importance_df.to_csv(output_path, index=False)
    logger.info(f"Feature importance data saved to {output_path}")
    
    # Create a simple bar plot of the top 15 features
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(15)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.splitext(output_path)[0] + '.png'
    save_plot(plt, plot_path)
    logger.info(f"Feature importance plot saved to {plot_path}")

def plot_model_metrics_table(
    metrics: Dict[str, Dict[str, float]],
    config: Dict[str, Any]
) -> None:
    """
    Generate a table comparing metrics across models.
    
    Args:
        metrics: Dictionary mapping model names to metrics
        config: Configuration dictionary
    """
    # Create DataFrame from metrics dictionary
    metrics_data = []
    
    for model_name, model_metrics in metrics.items():
        row = {'model': model_name}
        row.update(model_metrics)
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Save to CSV
    output_dir = config["paths"]["output_dir"]
    output_path = os.path.join(output_dir, "model_metrics_table.csv")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    metrics_df.to_csv(output_path, index=False)
    logger.info(f"Model metrics table saved to {output_path}")

def plot_r2_comparison(
    predictions: Dict[str, np.ndarray],
    target_values: np.ndarray,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate R² comparison data across models.
    
    Args:
        predictions: Dictionary mapping model names to predictions
        target_values: True target values
        model_names: List of model names
        config: Configuration dictionary
    """
    from sklearn.metrics import r2_score
    
    # Calculate R² for each model
    r2_data = []
    
    for model_name in model_names:
        if model_name in predictions:
            r2 = r2_score(target_values, predictions[model_name])
            r2_data.append({
                'model': model_name,
                'r2': r2
            })
    
    r2_df = pd.DataFrame(r2_data)
    
    # Save to CSV
    output_dir = config["paths"]["output_dir"]
    output_path = os.path.join(output_dir, "comparisons", "r2_comparison.csv")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    r2_df.to_csv(output_path, index=False)
    logger.info(f"R² comparison data saved to {output_path}")
    
    # Create a simple bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x='model', y='r2', data=r2_df)
    plt.title('R² Comparison Across Models')
    plt.ylim(0, 1)  # R² is typically between 0 and 1
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.splitext(output_path)[0] + '.png'
    save_plot(plt, plot_path)
    logger.info(f"R² comparison plot saved to {plot_path}")

def plot_residue_level_rmsf(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    target_col: str,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate residue-level RMSF comparison data.
    
    Args:
        df: DataFrame with protein data
        predictions: Dictionary mapping model names to predictions
        target_col: Target column name
        model_names: List of model names
        config: Configuration dictionary
    """
    # Sample a single domain for visualization
    domains = df['domain_id'].unique()
    if len(domains) > 0:
        # Select the domain with the most residues for better visualization
        domain_counts = df.groupby('domain_id').size()
        selected_domain = domain_counts.idxmax()
        
        domain_df = df[df['domain_id'] == selected_domain].copy()
        
        # Add predictions from each model
        for model_name in model_names:
            if model_name in predictions:
                domain_df[f"{model_name}_predicted"] = np.nan
                
                # Match predictions to domain rows
                for i, idx in enumerate(df.index):
                    if idx in domain_df.index:
                        domain_df.loc[idx, f"{model_name}_predicted"] = predictions[model_name][i]
        
        # Sort by residue ID
        domain_df = domain_df.sort_values('resid')
        
        # Select columns for output
        output_cols = ['domain_id', 'resid', 'resname', target_col]
        output_cols.extend([f"{model_name}_predicted" for model_name in model_names if model_name in predictions])
        
        if 'secondary_structure_encoded' in domain_df.columns:
            output_cols.append('secondary_structure_encoded')
        
        if 'core_exterior_encoded' in domain_df.columns:
            output_cols.append('core_exterior_encoded')
        
        # Save to CSV
        output_dir = config["paths"]["output_dir"]
        output_path = os.path.join(output_dir, "residue_analysis", f"domain_{selected_domain}_rmsf.csv")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        domain_df[output_cols].to_csv(output_path, index=False)
        logger.info(f"Residue-level RMSF data saved to {output_path}")
        
        # Create a simple line plot
        plt.figure(figsize=(12, 6))
        
        # Plot actual values
        plt.plot(domain_df['resid'], domain_df[target_col], 'k-', label='Actual', linewidth=2)
        
        # Plot predictions
        colors = make_model_color_map(model_names)
        for model_name in model_names:
            if model_name in predictions and f"{model_name}_predicted" in domain_df.columns:
                plt.plot(domain_df['resid'], domain_df[f"{model_name}_predicted"], 
                         label=model_name, color=colors.get(model_name), linewidth=1.5)
        
        plt.xlabel('Residue ID')
        plt.ylabel('RMSF')
        plt.title(f'RMSF Profile for Domain {selected_domain}')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.splitext(output_path)[0] + '.png'
        save_plot(plt, plot_path)
        logger.info(f"Residue-level RMSF plot saved to {plot_path}")

def plot_amino_acid_error_analysis(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    target_col: str,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate amino acid error analysis data.
    
    Args:
        df: DataFrame with protein data
        predictions: Dictionary mapping model names to predictions
        target_col: Target column name
        model_names: List of model names
        config: Configuration dictionary
    """
    if 'resname' not in df.columns:
        logger.warning("Residue name information not available for amino acid analysis")
        return
    
    # Calculate errors for each model
    df_with_preds = df.copy()
    
    for model_name in model_names:
        if model_name in predictions:
            df_with_preds[f"{model_name}_predicted"] = predictions[model_name]
            df_with_preds[f"{model_name}_error"] = predictions[model_name] - df_with_preds[target_col]
            df_with_preds[f"{model_name}_abs_error"] = np.abs(df_with_preds[f"{model_name}_error"])
    
    # Group by amino acid
    aa_errors = []
    
    for resname, group in df_with_preds.groupby('resname'):
        row = {'resname': resname, 'count': len(group)}
        
        for model_name in model_names:
            if model_name in predictions:
                error_col = f"{model_name}_abs_error"
                if error_col in group.columns:
                    row[f"{model_name}_mean_error"] = group[error_col].mean()
                    row[f"{model_name}_median_error"] = group[error_col].median()
                    row[f"{model_name}_std_error"] = group[error_col].std()
        
        aa_errors.append(row)
    
    aa_error_df = pd.DataFrame(aa_errors)
    
    # Save to CSV
    output_dir = config["paths"]["output_dir"]
    output_path = os.path.join(output_dir, "residue_analysis", "amino_acid_errors.csv")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    aa_error_df.to_csv(output_path, index=False)
    logger.info(f"Amino acid error analysis saved to {output_path}")
    
    # Create a simple bar plot for the model with the most predictions
    if model_names:
        model_name = model_names[0]  # Use first model
        
        if f"{model_name}_mean_error" in aa_error_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Sort by error
            sorted_df = aa_error_df.sort_values(f"{model_name}_mean_error")
            
            # Plot
            sns.barplot(x='resname', y=f"{model_name}_mean_error", data=sorted_df)
            plt.title(f'Mean Absolute Error by Amino Acid Type ({model_name})')
            plt.xlabel('Amino Acid')
            plt.ylabel('Mean Absolute Error')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.splitext(output_path)[0] + f"_{model_name}.png"
            save_plot(plt, plot_path)
            logger.info(f"Amino acid error plot saved to {plot_path}")

def plot_amino_acid_error_boxplot(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    target_col: str,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate amino acid error boxplot data.
    
    Args:
        df: DataFrame with protein data
        predictions: Dictionary mapping model names to predictions
        target_col: Target column name
        model_names: List of model names
        config: Configuration dictionary
    """
    if 'resname' not in df.columns or not model_names:
        return
    
    # Calculate errors for the first model
    model_name = model_names[0]
    
    if model_name in predictions:
        df_with_preds = df.copy()
        df_with_preds[f"{model_name}_predicted"] = predictions[model_name]
        df_with_preds[f"{model_name}_error"] = predictions[model_name] - df_with_preds[target_col]
        df_with_preds[f"{model_name}_abs_error"] = np.abs(df_with_preds[f"{model_name}_error"])
        
        # Create long-form data for boxplot
        error_data = []
        
        for _, row in df_with_preds.iterrows():
            error_data.append({
                'resname': row['resname'],
                'error': row[f"{model_name}_abs_error"]
            })
        
        error_df = pd.DataFrame(error_data)
        
        # Save to CSV
        output_dir = config["paths"]["output_dir"]
        output_path = os.path.join(output_dir, "residue_analysis", f"amino_acid_errors_boxplot_{model_name}.csv")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        error_df.to_csv(output_path, index=False)
        logger.info(f"Amino acid error boxplot data saved to {output_path}")
        
        # Create a boxplot
        plt.figure(figsize=(12, 6))
        
        # Get median errors for sorting
        median_errors = error_df.groupby('resname')['error'].median().sort_values()
        sorted_residues = median_errors.index.tolist()
        
        # Create boxplot with sorted residues
        sns.boxplot(x='resname', y='error', data=error_df, order=sorted_residues)
        plt.title(f'Absolute Error Distribution by Amino Acid Type ({model_name})')
        plt.xlabel('Amino Acid')
        plt.ylabel('Absolute Error')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.splitext(output_path)[0] + '.png'
        save_plot(plt, plot_path)
        logger.info(f"Amino acid error boxplot saved to {plot_path}")

def plot_amino_acid_scatter_plot(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    target_col: str,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate amino acid scatter plot data.
    
    Args:
        df: DataFrame with protein data
        predictions: Dictionary mapping model names to predictions
        target_col: Target column name
        model_names: List of model names
        config: Configuration dictionary
    """
    if 'resname' not in df.columns or not model_names:
        return
    
    # Use the first model for scatter plot
    model_name = model_names[0]
    
    if model_name in predictions:
        df_with_preds = df.copy()
        df_with_preds[f"{model_name}_predicted"] = predictions[model_name]
        
        # Create scatter plot data
        scatter_data = []
        
        for _, row in df_with_preds.iterrows():
            scatter_data.append({
                'resname': row['resname'],
                'actual': row[target_col],
                'predicted': row[f"{model_name}_predicted"]
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        # Save to CSV
        output_dir = config["paths"]["output_dir"]
        output_path = os.path.join(output_dir, "residue_analysis", f"amino_acid_scatter_{model_name}.csv")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        scatter_df.to_csv(output_path, index=False)
        logger.info(f"Amino acid scatter plot data saved to {output_path}")
        
        # Create a scatter plot
        plt.figure(figsize=(8, 8))
        
        # Sample up to 5000 points for better visibility
        sample_size = min(5000, len(scatter_df))
        sampled_df = scatter_df.sample(sample_size, random_state=config["system"]["random_state"])
        
        # Create a colormap for amino acids
        unique_residues = sampled_df['resname'].unique()
        cmap = plt.cm.get_cmap('tab20', len(unique_residues))
        residue_to_color = {res: cmap(i) for i, res in enumerate(unique_residues)}
        colors = [residue_to_color[res] for res in sampled_df['resname']]
        
        # Plot
        plt.scatter(sampled_df['actual'], sampled_df['predicted'], c=colors, alpha=0.7, s=30)
        
        # Add diagonal line
        min_val = min(sampled_df['actual'].min(), sampled_df['predicted'].min())
        max_val = max(sampled_df['actual'].max(), sampled_df['predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        plt.xlabel('Actual RMSF')
        plt.ylabel('Predicted RMSF')
        plt.title(f'Actual vs Predicted RMSF by Amino Acid Type ({model_name})')
        
        # Add legend with the most common amino acids (top 10)
        residue_counts = df['resname'].value_counts().head(10)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=residue_to_color[res], markersize=10, label=res) 
                         for res in residue_counts.index]
        plt.legend(handles=legend_elements, title='Amino Acid')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.splitext(output_path)[0] + '.png'
        save_plot(plt, plot_path)
        logger.info(f"Amino acid scatter plot saved to {plot_path}")

def plot_error_analysis_by_property(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    target_col: str,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate error analysis by property data.
    
    Args:
        df: DataFrame with protein data
        predictions: Dictionary mapping model names to predictions
        target_col: Target column name
        model_names: List of model names
        config: Configuration dictionary
    """
    if not model_names:
        return
    
    # Use the first model for analysis
    model_name = model_names[0]
    
    if model_name in predictions:
        df_with_preds = df.copy()
        df_with_preds[f"{model_name}_predicted"] = predictions[model_name]
        df_with_preds[f"{model_name}_error"] = predictions[model_name] - df_with_preds[target_col]
        df_with_preds[f"{model_name}_abs_error"] = np.abs(df_with_preds[f"{model_name}_error"])
        
        # Properties to analyze
        properties = []
        
        if 'secondary_structure_encoded' in df_with_preds.columns:
            properties.append({
                'name': 'secondary_structure',
                'column': 'secondary_structure_encoded',
                'labels': {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other'}
            })
        
        if 'core_exterior_encoded' in df_with_preds.columns:
            properties.append({
                'name': 'surface_exposure',
                'column': 'core_exterior_encoded',
                'labels': {0: 'Core', 1: 'Surface'}
            })
        
        if 'normalized_resid' in df_with_preds.columns:
            # Create bins for normalized position
            df_with_preds['position_bin'] = pd.cut(
                df_with_preds['normalized_resid'], 
                bins=5, 
                labels=['N-term', 'N-quarter', 'Middle', 'C-quarter', 'C-term']
            )
            
            properties.append({
                'name': 'sequence_position',
                'column': 'position_bin',
                'labels': None  # Use the bin labels
            })
        
        # Analyze each property
        for prop in properties:
            property_data = []
            
            if prop['column'] in df_with_preds.columns:
                groupby_col = prop['column']
                
                for group_val, group in df_with_preds.groupby(groupby_col):
                    # Get label
                    if prop['labels'] is not None:
                        label = prop['labels'].get(group_val, str(group_val))
                    else:
                        label = str(group_val)
                    
                    # Calculate metrics
                    row = {
                        'property': prop['name'],
                        'value': label,
                        'count': len(group),
                        'mean_error': group[f"{model_name}_abs_error"].mean(),
                        'median_error': group[f"{model_name}_abs_error"].median(),
                        'std_error': group[f"{model_name}_abs_error"].std()
                    }
                    
                    property_data.append(row)
                
                # Create DataFrame
                prop_df = pd.DataFrame(property_data)
                
                # Save to CSV
                output_dir = config["paths"]["output_dir"]
                output_path = os.path.join(
                    output_dir, 
                    "residue_analysis", 
                    f"error_by_{prop['name']}_{model_name}.csv"
                )
                
                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                prop_df.to_csv(output_path, index=False)
                logger.info(f"Error analysis by {prop['name']} saved to {output_path}")
                
                # Create a bar plot
                plt.figure(figsize=(8, 5))
                
                # Sort by value if not a categorical property
                if prop['name'] == 'sequence_position':
                    # Use the defined order for position bins
                    order = ['N-term', 'N-quarter', 'Middle', 'C-quarter', 'C-term']
                    sns.barplot(x='value', y='mean_error', data=prop_df, order=order)
                else:
                    sns.barplot(x='value', y='mean_error', data=prop_df)
                
                plt.title(f'Mean Absolute Error by {prop["name"].replace("_", " ").title()} ({model_name})')
                plt.xlabel(prop['name'].replace('_', ' ').title())
                plt.ylabel('Mean Absolute Error')
                plt.tight_layout()
                
                # Save the plot
                plot_path = os.path.splitext(output_path)[0] + '.png'
                save_plot(plt, plot_path)
                logger.info(f"Error analysis plot for {prop['name']} saved to {plot_path}")

def plot_r2_comparison_scatter(
    predictions: Dict[str, np.ndarray],
    target_values: np.ndarray,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate R² comparison scatter plot data.
    
    Args:
        predictions: Dictionary mapping model names to predictions
        target_values: True target values
        model_names: List of model names
        config: Configuration dictionary
    """
    if len(model_names) < 2:
        return
    
    # Get pairs of models
    model_pairs = []
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            if model1 in predictions and model2 in predictions:
                model_pairs.append((model1, model2))
    
    # Create scatter plot data for each pair
    for model1, model2 in model_pairs:
        scatter_data = []
        
        for i in range(len(target_values)):
            scatter_data.append({
                'actual': target_values[i],
                f"{model1}_predicted": predictions[model1][i],
                f"{model2}_predicted": predictions[model2][i]
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        # Save to CSV
        output_dir = config["paths"]["output_dir"]
        output_path = os.path.join(
            output_dir, 
            "comparisons", 
            f"scatter_{model1}_vs_{model2}.csv"
        )
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        scatter_df.to_csv(output_path, index=False)
        logger.info(f"Scatter plot data for {model1} vs {model2} saved to {output_path}")
        
        # Create a scatter plot
        plt.figure(figsize=(8, 8))
        
        # Sample up to 5000 points for better visibility
        sample_size = min(5000, len(scatter_df))
        sampled_df = scatter_df.sample(sample_size, random_state=config["system"]["random_state"])
        
        # Plot
        plt.scatter(
            sampled_df[f"{model1}_predicted"], 
            sampled_df[f"{model2}_predicted"], 
            c=sampled_df['actual'], 
            cmap='viridis', 
            alpha=0.7, 
            s=30
        )
        
        # Add diagonal line
        min_val = min(sampled_df[f"{model1}_predicted"].min(), sampled_df[f"{model2}_predicted"].min())
        max_val = max(sampled_df[f"{model1}_predicted"].max(), sampled_df[f"{model2}_predicted"].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        plt.xlabel(f'{model1} Predicted')
        plt.ylabel(f'{model2} Predicted')
        plt.title(f'Prediction Comparison: {model1} vs {model2}')
        plt.colorbar(label='Actual RMSF')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.splitext(output_path)[0] + '.png'
        save_plot(plt, plot_path)
        logger.info(f"Scatter plot for {model1} vs {model2} saved to {plot_path}")

def plot_scatter_with_density_contours(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    target_col: str,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate scatter plot with density contours.
    
    Args:
        df: DataFrame with protein data
        predictions: Dictionary mapping model names to predictions
        target_col: Target column name
        model_names: List of model names
        config: Configuration dictionary
    """
    if not model_names:
        return
    
    # Use first model
    model_name = model_names[0]
    
    if model_name in predictions:
        # Create data for plotting
        scatter_data = []
        
        for i, idx in enumerate(df.index):
            scatter_data.append({
                'actual': df.loc[idx, target_col],
                'predicted': predictions[model_name][i]
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        # Save to CSV
        output_dir = config["paths"]["output_dir"]
        output_path = os.path.join(
            output_dir, 
            "comparisons", 
            f"density_scatter_{model_name}.csv"
        )
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        scatter_df.to_csv(output_path, index=False)
        logger.info(f"Density scatter plot data for {model_name} saved to {output_path}")
        
        # Create the plot
        plt.figure(figsize=(8, 8))
        
        # Sample up to 5000 points for better visibility
        sample_size = min(5000, len(scatter_df))
        sampled_df = scatter_df.sample(sample_size, random_state=config["system"]["random_state"])
        
        # Create plot with density contours
        sns.kdeplot(
            x='actual',
            y='predicted',
            data=sampled_df,
            fill=True,
            cmap='Blues',
            alpha=0.5,
            levels=10
        )
        
        plt.scatter(
            sampled_df['actual'],
            sampled_df['predicted'],
            alpha=0.3,
            s=20,
            c='darkblue'
        )
        
        # Add diagonal line
        min_val = min(sampled_df['actual'].min(), sampled_df['predicted'].min())
        max_val = max(sampled_df['actual'].max(), sampled_df['predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual RMSF')
        plt.ylabel('Predicted RMSF')
        plt.title(f'Actual vs Predicted RMSF with Density Contours ({model_name})')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.splitext(output_path)[0] + '.png'
        save_plot(plt, plot_path)
        logger.info(f"Density scatter plot for {model_name} saved to {plot_path}")

def plot_flexibility_vs_dihedral_angles(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    target_col: str,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate flexibility vs dihedral angles plot data.
    
    Args:
        df: DataFrame with protein data
        predictions: Dictionary mapping model names to predictions
        target_col: Target column name
        model_names: List of model names
        config: Configuration dictionary
    """
    if 'phi_norm' not in df.columns or 'psi_norm' not in df.columns:
        return
    
    # Use first model
    model_name = model_names[0] if model_names else None
    
    # Prepare data for plotting
    plot_data = []
    
    for i, idx in enumerate(df.index):
        row = {
            'phi': df.loc[idx, 'phi_norm'],
            'psi': df.loc[idx, 'psi_norm'],
            'actual': df.loc[idx, target_col]
        }
        
        if model_name and model_name in predictions:
            row['predicted'] = predictions[model_name][i]
        
        plot_data.append(row)
    
    plot_df = pd.DataFrame(plot_data)
    
    # Save to CSV
    output_dir = config["paths"]["output_dir"]
    output_path = os.path.join(
        output_dir, 
        "residue_analysis", 
        "flexibility_vs_dihedral_angles.csv"
    )
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plot_df.to_csv(output_path, index=False)
    logger.info(f"Flexibility vs dihedral angles data saved to {output_path}")
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Sample up to 5000 points for better visibility
    sample_size = min(5000, len(plot_df))
    sampled_df = plot_df.sample(sample_size, random_state=config["system"]["random_state"])
    
    # Create heatmap scatter plot
    plt.scatter(
        sampled_df['phi'],
        sampled_df['psi'],
        c=sampled_df['actual'],
        cmap='viridis',
        alpha=0.7,
        s=30
    )
    
    plt.xlabel('Normalized Phi Angle')
    plt.ylabel('Normalized Psi Angle')
    plt.title('Protein Flexibility in Dihedral Angle Space')
    plt.colorbar(label='RMSF')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.splitext(output_path)[0] + '.png'
    save_plot(plt, plot_path)
    logger.info(f"Flexibility vs dihedral angles plot saved to {plot_path}")

def plot_flexibility_sequence_neighborhood(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    target_col: str,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate flexibility vs sequence neighborhood plot data.
    
    Args:
        df: DataFrame with protein data
        predictions: Dictionary mapping model names to predictions
        target_col: Target column name
        model_names: List of model names
        config: Configuration dictionary
    """
    # Check if we have window features
    window_cols = [col for col in df.columns if '_offset_' in col]
    
    if not window_cols:
        return
    
    # Find a specific domain with good data
    domains = df['domain_id'].unique()
    
    if len(domains) > 0:
        # Select a domain with at least 50 residues
        domain_sizes = df.groupby('domain_id').size()
        valid_domains = domain_sizes[domain_sizes >= 50].index
        
        if len(valid_domains) > 0:
            selected_domain = valid_domains[0]
            domain_df = df[df['domain_id'] == selected_domain].copy()
            
            # Sort by residue ID
            domain_df = domain_df.sort_values('resid')
            
            # Select a window around a highly flexible residue
            flexible_idx = domain_df[target_col].idxmax()
            flexible_resid = domain_df.loc[flexible_idx, 'resid']
            
            # Select residues within 10 positions
            window_size = 10
            min_resid = max(0, flexible_resid - window_size)
            max_resid = flexible_resid + window_size
            
            window_df = domain_df[
                (domain_df['resid'] >= min_resid) & 
                (domain_df['resid'] <= max_resid)
            ].copy()
            
            # Add predictions if available
            if model_names and model_names[0] in predictions:
                model_name = model_names[0]
                window_df[f"{model_name}_predicted"] = np.nan
                
                # Match predictions to domain rows
                for i, idx in enumerate(df.index):
                    if idx in window_df.index:
                        window_df.loc[idx, f"{model_name}_predicted"] = predictions[model_name][i]
            
            # Save to CSV
            output_dir = config["paths"]["output_dir"]
            output_path = os.path.join(
                output_dir, 
                "residue_analysis", 
                f"sequence_neighborhood_domain_{selected_domain}.csv"
            )
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Select relevant columns
            relevant_cols = ['domain_id', 'resid', 'resname', target_col, 'secondary_structure_encoded']
            
            if model_names and model_names[0] in predictions:
                model_name = model_names[0]
                if f"{model_name}_predicted" in window_df.columns:
                    relevant_cols.append(f"{model_name}_predicted")
            
            window_df[relevant_cols].to_csv(output_path, index=False)
            logger.info(f"Sequence neighborhood data saved to {output_path}")
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            # Plot actual values
            plt.plot(
                window_df['resid'], 
                window_df[target_col], 
                'k-', 
                linewidth=2, 
                label='Actual'
            )
            
            # Plot predictions if available
            if model_names and model_names[0] in predictions:
                model_name = model_names[0]
                if f"{model_name}_predicted" in window_df.columns:
                    plt.plot(
                        window_df['resid'], 
                        window_df[f"{model_name}_predicted"], 
                        'r--', 
                        linewidth=1.5, 
                        label='Predicted'
                    )
            
            # Add secondary structure if available
            if 'secondary_structure_encoded' in window_df.columns:
                # Create secondary structure bars at the bottom
                for i, row in window_df.iterrows():
                    ss = row['secondary_structure_encoded']
                    resid = row['resid']
                    
                    if ss == 0:  # Helix
                        plt.axvspan(resid-0.4, resid+0.4, alpha=0.2, color='red', ymin=0, ymax=0.05)
                    elif ss == 1:  # Sheet
                        plt.axvspan(resid-0.4, resid+0.4, alpha=0.2, color='blue', ymin=0, ymax=0.05)
                    else:  # Loop/Other
                        plt.axvspan(resid-0.4, resid+0.4, alpha=0.2, color='green', ymin=0, ymax=0.05)
                
                # Add legend for secondary structure
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', alpha=0.2, label='Helix'),
                    Patch(facecolor='blue', alpha=0.2, label='Sheet'),
                    Patch(facecolor='green', alpha=0.2, label='Loop/Other')
                ]
                
                plt.legend(handles=legend_elements, loc='upper right')
            
            plt.xlabel('Residue ID')
            plt.ylabel('RMSF')
            plt.title(f'Flexibility in Sequence Neighborhood (Domain {selected_domain})')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.splitext(output_path)[0] + '.png'
            save_plot(plt, plot_path)
            logger.info(f"Sequence neighborhood plot saved to {plot_path}")

def plot_error_response_surface(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    target_col: str,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate error response surface plot data.
    
    Args:
        df: DataFrame with protein data
        predictions: Dictionary mapping model names to predictions
        target_col: Target column name
        model_names: List of model names
        config: Configuration dictionary
    """
    if not model_names or 'normalized_resid' not in df.columns:
        return
    
    # Use first model
    model_name = model_names[0]
    
    if model_name in predictions:
        # Calculate errors
        df_with_preds = df.copy()
        df_with_preds[f"{model_name}_predicted"] = predictions[model_name]
        df_with_preds[f"{model_name}_error"] = predictions[model_name] - df_with_preds[target_col]
        df_with_preds[f"{model_name}_abs_error"] = np.abs(df_with_preds[f"{model_name}_error"])
        
        # Create bins for normalized position and secondary structure
        if 'secondary_structure_encoded' in df_with_preds.columns:
            # Create data for heatmap
            heatmap_data = []
            
            # Create position bins
            df_with_preds['position_bin'] = pd.cut(
                df_with_preds['normalized_resid'], 
                bins=10, 
                labels=range(10)
            )
            
            # Group by position bin and secondary structure
            grouped = df_with_preds.groupby(['position_bin', 'secondary_structure_encoded'])
            
            for (pos_bin, ss), group in grouped:
                heatmap_data.append({
                    'position_bin': pos_bin,
                    'secondary_structure': ss,
                    'mean_error': group[f"{model_name}_abs_error"].mean(),
                    'count': len(group)
                })
            
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Save to CSV
            output_dir = config["paths"]["output_dir"]
            output_path = os.path.join(
                output_dir, 
                "residue_analysis", 
                f"error_response_surface_{model_name}.csv"
            )
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            heatmap_df.to_csv(output_path, index=False)
            logger.info(f"Error response surface data saved to {output_path}")
            
            # Create a pivot table for the heatmap
            pivot_df = heatmap_df.pivot(
                index='secondary_structure', 
                columns='position_bin', 
                values='mean_error'
            )
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            # Create heatmap
            sns.heatmap(
                pivot_df, 
                cmap='viridis', 
                annot=True, 
                fmt=".2f", 
                linewidths=0.5
            )
            
            # Set labels
            plt.xlabel('Normalized Position Bin')
            plt.ylabel('Secondary Structure (0=Helix, 1=Sheet, 2=Loop)')
            plt.title(f'Error Response Surface: Position vs Structure ({model_name})')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.splitext(output_path)[0] + '.png'
            save_plot(plt, plot_path)
            logger.info(f"Error response surface plot saved to {plot_path}")

def plot_secondary_structure_error_correlation(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    target_col: str,
    model_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Generate secondary structure error correlation plot data.
    
    Args:
        df: DataFrame with protein data
        predictions: Dictionary mapping model names to predictions
        target_col: Target column name
        model_names: List of model names
        config: Configuration dictionary
    """
    if 'secondary_structure_encoded' not in df.columns or not model_names:
        return
    
    # Use first model
    model_name = model_names[0]
    
    if model_name in predictions:
        # Calculate errors
        df_with_preds = df.copy()
        df_with_preds[f"{model_name}_predicted"] = predictions[model_name]
        df_with_preds[f"{model_name}_error"] = predictions[model_name] - df_with_preds[target_col]
        df_with_preds[f"{model_name}_abs_error"] = np.abs(df_with_preds[f"{model_name}_error"])
        
        # Group by secondary structure
        ss_errors = []
        
        for ss, group in df_with_preds.groupby('secondary_structure_encoded'):
            ss_name = {0: 'Helix', 1: 'Sheet', 2: 'Loop/Other'}.get(ss, str(ss))
            
            # Calculate metrics
            row = {
                'secondary_structure': ss_name,
                'count': len(group),
                'mean_actual': group[target_col].mean(),
                'mean_predicted': group[f"{model_name}_predicted"].mean(),
                'mean_error': group[f"{model_name}_abs_error"].mean(),
                'std_error': group[f"{model_name}_abs_error"].std()
            }
            
            ss_errors.append(row)
        
        ss_error_df = pd.DataFrame(ss_errors)
        
        # Save to CSV
        output_dir = config["paths"]["output_dir"]
        output_path = os.path.join(
            output_dir, 
            "residue_analysis", 
            f"secondary_structure_errors_{model_name}.csv"
        )
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        ss_error_df.to_csv(output_path, index=False)
        logger.info(f"Secondary structure error correlation data saved to {output_path}")
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        x = np.arange(len(ss_error_df))
        width = 0.35
        
        plt.bar(
            x - width/2, 
            ss_error_df['mean_actual'], 
            width, 
            label='Actual RMSF'
        )
        
        plt.bar(
            x + width/2, 
            ss_error_df['mean_predicted'], 
            width, 
            label='Predicted RMSF'
        )
        
        plt.xlabel('Secondary Structure')
        plt.ylabel('Mean RMSF')
        plt.title(f'Actual vs Predicted RMSF by Secondary Structure ({model_name})')
        plt.xticks(x, ss_error_df['secondary_structure'])
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.splitext(output_path)[0] + '.png'
        save_plot(plt, plot_path)
        logger.info(f"Secondary structure comparison plot saved to {plot_path}")

def plot_training_validation_curves(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    model_name: str,
    config: Dict[str, Any]
) -> None:
    """
    Generate training and validation curves.
    
    Args:
        train_metrics: Dictionary of training metrics by epoch
        val_metrics: Dictionary of validation metrics by epoch
        model_name: Name of the model
        config: Configuration dictionary
    """
    if not train_metrics or not val_metrics:
        return
    
    # Convert to DataFrame
    epochs = len(train_metrics.get('train_loss', []))
    
    if epochs == 0:
        return
    
    curve_data = []
    
    for i in range(epochs):
        row = {'epoch': i}
        
        for metric, values in train_metrics.items():
            if i < len(values):
                metric_name = metric.replace('train_', '')
                row[f"train_{metric_name}"] = values[i]
        
        for metric, values in val_metrics.items():
            if i < len(values):
                metric_name = metric.replace('val_', '')
                row[f"val_{metric_name}"] = values[i]
        
        curve_data.append(row)
    
    curve_df = pd.DataFrame(curve_data)
    
    # Save to CSV
    output_dir = config["paths"]["output_dir"]
    output_path = os.path.join(
        output_dir, 
        "training_performance", 
        f"{model_name}_training_curves.csv"
    )
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    curve_df.to_csv(output_path, index=False)
    logger.info(f"Training and validation curves data saved to {output_path}")
    
    # Create the plots
    metrics_to_plot = []
    
    if 'train_loss' in train_metrics and 'val_loss' in val_metrics:
        metrics_to_plot.append(('loss', 'Loss'))
    
    if 'train_r2' in train_metrics and 'val_r2' in val_metrics:
        metrics_to_plot.append(('r2', 'R²'))
    
    for metric, title in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        plt.plot(
            curve_df['epoch'], 
            curve_df[f"train_{metric}"], 
            'b-', 
            label=f'Training {title}'
        )
        
        plt.plot(
            curve_df['epoch'], 
            curve_df[f"val_{metric}"], 
            'r-', 
            label=f'Validation {title}'
        )
        
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(f'{title} Curves for {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(
            output_dir, 
            "training_performance", 
            f"{model_name}_{metric}_curves.png"
        )
        
        save_plot(plt, plot_path)
        logger.info(f"{title} curves plot saved to {plot_path}")

def plot_neural_network_learning_dynamics(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    model_name: str,
    config: Dict[str, Any]
) -> None:
    """
    Generate neural network learning dynamics visualization.
    
    Args:
        train_metrics: Dictionary of training metrics by epoch
        val_metrics: Dictionary of validation metrics by epoch
        model_name: Name of the model
        config: Configuration dictionary
    """
    if not train_metrics or not val_metrics:
        return
    
    # Convert to DataFrame
    epochs = len(train_metrics.get('train_loss', []))
    
    if epochs == 0:
        return
    
    # Create a combined plot with multiple metrics
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    if 'train_loss' in train_metrics and 'val_loss' in val_metrics:
        ax1 = plt.subplot(2, 1, 1)
        
        # Plot loss
        ax1.plot(
            range(epochs), 
            train_metrics['train_loss'], 
            'b-', 
            label='Training Loss'
        )
        
        ax1.plot(
            range(epochs), 
            val_metrics['val_loss'], 
            'r-', 
            label='Validation Loss'
        )
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Loss Curves for {model_name}')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    if 'train_r2' in train_metrics and 'val_r2' in val_metrics:
        ax2 = plt.subplot(2, 1, 2)
        
        # Plot R²
        ax2.plot(
            range(epochs), 
            train_metrics['train_r2'], 
            'g-', 
            label='Training R²'
        )
        
        ax2.plot(
            range(epochs), 
            val_metrics['val_r2'], 
            'y-', 
            label='Validation R²'
        )
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R²')
        ax2.set_title(f'R² Curves for {model_name}')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = config["paths"]["output_dir"]
    plot_path = os.path.join(
        output_dir, 
        "training_performance", 
        f"{model_name}_learning_dynamics.png"
    )
    
    # Create output directory
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    save_plot(plt, plot_path)
    logger.info(f"Learning dynamics plot saved to {plot_path}")