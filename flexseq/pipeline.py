"""
Main pipeline orchestration for the FlexSeq ML workflow.

This module provides the Pipeline class that handles the entire ML workflow
from data loading to evaluation, with temperature-specific functionality.
"""

import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
import numpy as np
import joblib

from flexseq.config import (
    load_config, 
    get_enabled_models, 
    get_model_config,
    get_output_dir_for_temperature
)
from flexseq.utils.helpers import progress_bar, ProgressCallback
from flexseq.models import get_model_class
from flexseq.data.processor import (
    load_and_process_data, 
    split_data, 
    prepare_data_for_model,
    process_features
)
from flexseq.utils.metrics import evaluate_predictions


logger = logging.getLogger(__name__)

class Pipeline:
    """
    Main pipeline orchestration for FlexSeq.
    Handles the full ML workflow from data loading to evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        
        # Create output directories if they don't exist
        self.prepare_directories()
        
        # Log mode information
        mode = config["mode"]["active"]
        logger.info(f"Pipeline initialized in {mode.upper()} mode")
        
        # Log temperature information
        temperature = config["temperature"]["current"]
        logger.info(f"Using temperature: {temperature}")
        
    def prepare_directories(self) -> None:
        """
        Create necessary output directories.
        """
        paths = self.config["paths"]
        
        # Ensure data directory exists
        data_dir = paths.get("data_dir", "./data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Ensure output directory exists
        output_dir = paths.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure models directory exists
        models_dir = paths.get("models_dir", "./models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Create subdirectories for different types of output
        os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "feature_importance"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "residue_analysis"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "domain_analysis"), exist_ok=True)
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and process input data.
        
        Args:
            data_path: Optional explicit path to data file
            
        Returns:
            Processed DataFrame
        """
        # Use explicit file path or temperature-specific data
        return load_and_process_data(data_path, self.config)
    
    def train(
        self, 
        model_names: Optional[List[str]] = None,
        data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train specified models on the data.
        
        Args:
            model_names: Optional list of model names to train (if None, use all enabled models)
            data_path: Optional explicit path to data file
            
        Returns:
            Dictionary of trained models
        """
        from flexseq.utils.helpers import progress_bar
        
        # Determine which models to train
        if model_names is None:
            model_names = get_enabled_models(self.config)
            
        if not model_names:
            logger.warning("No models specified for training")
            return {}
            
        # Load and preprocess data
        logger.info("Loading and processing data")
        with ProgressCallback(total=1, desc="Loading data") as pbar:
            df = self.load_data(data_path)
            pbar.update()
        
        # Split data
        logger.info("Splitting data into train/validation/test sets")
        with ProgressCallback(total=1, desc="Splitting data") as pbar:
            train_df, val_df, test_df = split_data(df, self.config)
            pbar.update()
        
        # Prepare training data
        with ProgressCallback(total=1, desc="Preparing features") as pbar:
            X_train, y_train, feature_names = prepare_data_for_model(train_df, self.config)
            pbar.update()
        
        # Train each model
        trained_models = {}
        
        for model_name in progress_bar(model_names, desc="Training models"):
            logger.info(f"Training model: {model_name}")
            
            try:
                # Get model class and config
                model_class = get_model_class(model_name)
                model_config = get_model_config(self.config, model_name)
                
                if not model_config.get("enabled", False):
                    logger.warning(f"Model {model_name} is disabled in config. Skipping.")
                    continue
                
                # Get hyperparameter optimization config
                optimize_hyperparams = False
                if model_name == "neural_network":
                    optimize_hyperparams = model_config.get("hyperparameter_optimization", {}).get("enabled", False)
                elif model_name == "random_forest":
                    optimize_hyperparams = model_config.get("randomized_search", {}).get("enabled", False)
                
                # Remove non-init params from config
                if model_name == "neural_network":
                    init_params = {
                        "architecture": model_config.get("architecture", {}),
                        "training": model_config.get("training", {}),
                        "random_state": self.config["system"].get("random_state", 42)
                    }
                else:
                    init_params = {k: v for k, v in model_config.items() 
                                if k not in ['enabled', 'cross_validation', 'save_best', 'randomized_search', 'hyperparameter_optimization']}
                
                # Create model instance
                model = model_class(**init_params)
                
                # Perform hyperparameter optimization if enabled
                if optimize_hyperparams:
                    with ProgressCallback(total=1, desc=f"Optimizing hyperparameters for {model_name}") as pbar:
                        logger.info(f"Performing hyperparameter optimization for {model_name}")
                        
                        if model_name == "neural_network":
                            opt_config = model_config["hyperparameter_optimization"]
                            method = opt_config.get("method", "bayesian")
                            trials = opt_config.get("trials", 20)
                            param_grid = opt_config.get("parameters", {})
                            
                            # Prepare validation data for hyperparameter tuning
                            X_val, y_val, _ = prepare_data_for_model(val_df, self.config)
                            
                            # Combine train and validation for cross-validation
                            X_combined = np.vstack([X_train, X_val])
                            y_combined = np.concatenate([y_train, y_val])
                            
                            # Optimize hyperparameters
                            best_params = model.hyperparameter_optimize(
                                X_combined, y_combined, param_grid, method, trials, cv=3
                            )
                            
                            logger.info(f"Best hyperparameters for {model_name}: {best_params}")
                            
                        elif model_name == "random_forest":
                            # Random forest uses its internal RandomizedSearchCV
                            # Just log that optimization is enabled
                            logger.info("RandomizedSearchCV will be used for Random Forest training")
                        
                        pbar.update()
                
                # Train the model
                start_time = time.time()
                
                if model_name == "neural_network":
                    # Neural network training shows progress
                    model.fit(X_train, y_train, feature_names)
                else:
                    # Other models use simple progress indicator
                    with ProgressCallback(total=1, desc=f"Training {model_name}") as pbar:
                        model.fit(X_train, y_train, feature_names)
                        pbar.update()
                
                # Store trained model
                trained_models[model_name] = model
                
                # Log training time
                train_time = time.time() - start_time
                logger.info(f"Trained {model_name} in {train_time:.2f} seconds")
                
                # Save training history if available
                if hasattr(model, 'get_training_history') and model.get_training_history():
                    history = model.get_training_history()
                    history_df = pd.DataFrame(history)
                    history_path = os.path.join(self.config["paths"]["output_dir"], f"{model_name}_training_history.csv")
                    history_df.to_csv(history_path)
                    logger.info(f"Saved training history to {history_path}")
                
                # Save model if configured
                if model_config.get("save_best", True):
                    with ProgressCallback(total=1, desc=f"Saving {model_name}") as pbar:
                        self.save_model(model, model_name)
                        pbar.update()
                
                # Evaluate on validation set
                with ProgressCallback(total=1, desc=f"Validating {model_name}") as pbar:
                    X_val, y_val, _ = prepare_data_for_model(val_df, self.config)
                    val_predictions = model.predict(X_val)
                    val_metrics = evaluate_predictions(y_val, val_predictions, self.config)
                    logger.info(f"Validation metrics for {model_name}: {val_metrics}")
                    pbar.update()
                
            except Exception as e:
                logger.error(f"Error training model {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        self.models = trained_models
        return trained_models

    
    def save_model(self, model: Any, model_name: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model instance
            model_name: Name of the model
        """
        models_dir = self.config["paths"]["models_dir"]
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        
        try:
            model.save(model_path)
            logger.info(f"Saved model {model_name} to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model instance
        """
        models_dir = self.config["paths"]["models_dir"]
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model_class = get_model_class(model_name)
            model = model_class.load(model_path)
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
        
    
    def evaluate(
        self, 
        model_names: Optional[List[str]] = None,
        data_path: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate models on test data.
        
        Args:
            model_names: Optional list of model names to evaluate
            data_path: Optional explicit path to data file
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        from flexseq.utils.helpers import progress_bar, ProgressCallback
        
        # Determine which models to evaluate
        if model_names is None:
            model_names = list(self.models.keys())
            
            # If no models in memory, use enabled models from config
            if not model_names:
                model_names = get_enabled_models(self.config)
                
        if not model_names:
            logger.warning("No models specified for evaluation")
            return {}
        
        # Load data if needed
        with ProgressCallback(total=1, desc="Loading data") as pbar:
            df = self.load_data(data_path)
            pbar.update()
        
        # Split data
        with ProgressCallback(total=1, desc="Splitting data") as pbar:
            train_df, val_df, test_df = split_data(df, self.config)
            pbar.update()
        
        # Use test or validation set based on config
        comparison_set = self.config["evaluation"]["comparison_set"]
        
        if comparison_set == "test":
            eval_df = test_df
            logger.info("Using test set for evaluation")
        elif comparison_set == "validation":
            eval_df = val_df
            logger.info("Using validation set for evaluation")
        else:
            logger.warning(f"Unknown comparison_set '{comparison_set}', using test set")
            eval_df = test_df
        
        # Prepare evaluation data
        with ProgressCallback(total=1, desc="Preparing features") as pbar:
            X_eval, y_eval, feature_names = prepare_data_for_model(eval_df, self.config)
            pbar.update()
        
        # Evaluate each model
        results = {}
        predictions = {}
        uncertainties = {}
        
        for model_name in progress_bar(model_names, desc="Evaluating models"):
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Load model if not in memory
                with ProgressCallback(total=1, desc=f"Loading {model_name}", leave=False) as pbar:
                    if model_name in self.models:
                        model = self.models[model_name]
                    else:
                        model = self.load_model(model_name)
                    pbar.update()
                
                # Generate predictions
                with ProgressCallback(total=1, desc=f"Predicting with {model_name}", leave=False) as pbar:
                    # Try to get uncertainty estimates if available
                    if hasattr(model, 'predict_with_std'):
                        preds, stds = model.predict_with_std(X_eval)
                        predictions[model_name] = preds
                        uncertainties[model_name] = stds
                    else:
                        preds = model.predict(X_eval)
                        predictions[model_name] = preds
                    pbar.update()
                
                # Calculate metrics
                with ProgressCallback(total=1, desc=f"Computing metrics", leave=False) as pbar:
                    metrics = evaluate_predictions(y_eval, preds, self.config, X_eval, X_eval.shape[1])
                    pbar.update()
                
                # Store results
                results[model_name] = metrics
                
                # Log results
                logger.info(f"Evaluation metrics for {model_name}: {metrics}")
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
        
        # Save evaluation results
        with ProgressCallback(total=1, desc="Saving evaluation results") as pbar:
            self.save_evaluation_results(results, eval_df, predictions, uncertainties)
            pbar.update()
            
        
        
        return results
    
    def save_evaluation_results(
        self, 
        results: Dict[str, Dict[str, float]],
        eval_df: pd.DataFrame,
        predictions: Dict[str, np.ndarray] = None,
        uncertainties: Dict[str, np.ndarray] = None
    ) -> None:
        """
        Save evaluation results to disk.
        
        Args:
            results: Dictionary of evaluation metrics for each model
            eval_df: DataFrame with evaluation data
            predictions: Dictionary of predictions by model
            uncertainties: Dictionary of prediction uncertainties by model
        """
        output_dir = self.config["paths"]["output_dir"]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to CSV
        results_path = os.path.join(output_dir, "evaluation_results.csv")
        results_df = pd.DataFrame(results).T
        results_df.index.name = "model"
        results_df.to_csv(results_path)
        logger.info(f"Saved evaluation metrics to {results_path}")
        
        # Save all results (predictions and optionally uncertainties)
        if predictions:
            all_results_df = eval_df.copy()
            target_col = self.config["dataset"]["target"]
            
            # Add predictions for each model
            for model_name, preds in predictions.items():
                all_results_df[f"{model_name}_predicted"] = preds
                
                # Add errors
                all_results_df[f"{model_name}_error"] = preds - all_results_df[target_col]
                all_results_df[f"{model_name}_abs_error"] = np.abs(all_results_df[f"{model_name}_error"])
                
                # Add uncertainties if available
                if uncertainties and model_name in uncertainties:
                    all_results_df[f"{model_name}_uncertainty"] = uncertainties[model_name]
            
            # Save to CSV
            all_results_path = os.path.join(output_dir, "all_results.csv")
            all_results_df.to_csv(all_results_path, index=False)
            logger.info(f"Saved detailed results to {all_results_path}")
            
            # Save domain-level metrics
            self.save_domain_metrics(all_results_df, target_col, predictions.keys())
    
    def save_domain_metrics(
        self,
        results_df: pd.DataFrame,
        target_col: str,
        model_names: List[str]
    ) -> None:
        """
        Calculate and save domain-level metrics.
        
        Args:
            results_df: DataFrame with all results
            target_col: Target column name
            model_names: List of model names
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        output_dir = self.config["paths"]["output_dir"]
        domain_metrics = []
        
        # Calculate metrics per domain
        for domain_id, domain_df in results_df.groupby("domain_id"):
            domain_result = {"domain_id": domain_id}
            
            # Calculate metrics for each model
            for model_name in model_names:
                pred_col = f"{model_name}_predicted"
                if pred_col not in domain_df.columns:
                    continue
                
                actual = domain_df[target_col].values
                predicted = domain_df[pred_col].values
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mae = mean_absolute_error(actual, predicted)
                r2 = r2_score(actual, predicted)
                
                # Store metrics
                domain_result[f"{model_name}_rmse"] = rmse
                domain_result[f"{model_name}_mae"] = mae
                domain_result[f"{model_name}_r2"] = r2
                
                # Basic statistics
                domain_result[f"{model_name}_mean_error"] = np.mean(domain_df[f"{model_name}_error"])
                domain_result[f"{model_name}_std_error"] = np.std(domain_df[f"{model_name}_error"])
                
                # Calculate temperature-dependent metrics if in OmniFlex mode
                if self.config["mode"]["active"] == "omniflex":
                    # Add correlation with ESM and voxel predictions if available
                    for pred_type in ["esm_rmsf", "voxel_rmsf"]:
                        if pred_type in domain_df.columns:
                            corr = np.corrcoef(predicted, domain_df[pred_type])[0, 1]
                            domain_result[f"{model_name}_corr_with_{pred_type}"] = corr
            
            # Add domain properties
            domain_result["num_residues"] = len(domain_df)
            
            # Calculate protein properties if available
            if "core_exterior_encoded" in domain_df.columns:
                domain_result["percent_surface"] = domain_df["core_exterior_encoded"].mean() * 100
                
            if "secondary_structure_encoded" in domain_df.columns:
                # Count residues by structure type
                ss_counts = domain_df["secondary_structure_encoded"].value_counts(normalize=True) * 100
                for ss_type, value in zip([0, 1, 2], ["helix", "sheet", "loop"]):
                    if ss_type in ss_counts:
                        domain_result[f"percent_{value}"] = ss_counts[ss_type]
                    else:
                        domain_result[f"percent_{value}"] = 0.0
                    
            domain_metrics.append(domain_result)
        
        # Save domain metrics to CSV
        if domain_metrics:
            domain_metrics_df = pd.DataFrame(domain_metrics)
            domain_metrics_path = os.path.join(output_dir, "domain_metrics.csv")
            domain_metrics_df.to_csv(domain_metrics_path, index=False)
            logger.info(f"Saved domain-level metrics to {domain_metrics_path}")

    def predict(
        self, 
        data: Union[str, pd.DataFrame],
        model_name: Optional[str] = None,
        with_uncertainty: bool = False
    ) -> pd.DataFrame:
        """
        Generate predictions for new data.
        
        Args:
            data: DataFrame or path to CSV file with protein data
            model_name: Model to use for prediction (if None, use best model)
            with_uncertainty: Whether to include uncertainty estimates
            
        Returns:
            DataFrame with original data and predictions
        """
        # Load data if string path provided
        if isinstance(data, str):
            df = load_and_process_data(data, self.config)
        else:
            df = data.copy()
            df = process_features(df, self.config)
        
        # Determine which model to use
        if model_name is None:
            # Find best model based on previous evaluation
            try:
                output_dir = self.config["paths"]["output_dir"]
                results_path = os.path.join(output_dir, "evaluation_results.csv")
                
                if os.path.exists(results_path):
                    results_df = pd.read_csv(results_path, index_col="model")
                    
                    # Use R^2 or RMSE to determine best model
                    if "r2" in results_df.columns:
                        best_model = results_df["r2"].idxmax()
                    elif "rmse" in results_df.columns:
                        best_model = results_df["rmse"].idxmin()
                    else:
                        best_model = results_df.index[0]
                        
                    model_name = best_model
                    logger.info(f"Using best model based on evaluation: {model_name}")
                else:
                    # No evaluation results, use first enabled model
                    model_name = get_enabled_models(self.config)[0]
                    logger.info(f"No evaluation results found, using enabled model: {model_name}")
                    
            except Exception as e:
                logger.error(f"Error finding best model: {e}")
                # Fall back to first enabled model
                model_name = get_enabled_models(self.config)[0]
        
        # Load model if not in memory
        if model_name in self.models:
            model = self.models[model_name]
        else:
            model = self.load_model(model_name)
        
        # Prepare data for prediction
        X, _, feature_names = prepare_data_for_model(
            df, self.config, include_target=False
        )
        
        # Generate predictions, possibly with uncertainty
        target_col = self.config["dataset"]["target"]
        result_df = df.copy()
        
        if with_uncertainty and hasattr(model, 'predict_with_std'):
            try:
                predictions, uncertainties = model.predict_with_std(X)
                
                # Add predictions and uncertainties to result
                result_df[f"{target_col}_predicted"] = predictions
                result_df[f"{target_col}_uncertainty"] = uncertainties
                
                # If in OmniFlex mode, add prediction quality indicators
                if self.config["mode"]["active"] == "omniflex":
                    # Calculate z-scores (deviation / uncertainty)
                    if target_col in result_df.columns:
                        z_scores = np.abs(predictions - result_df[target_col]) / uncertainties
                        result_df[f"{target_col}_z_score"] = z_scores
                
            except Exception as e:
                logger.error(f"Error generating predictions with uncertainty: {e}")
                # Fall back to standard prediction
                predictions = model.predict(X)
                result_df[f"{target_col}_predicted"] = predictions
        else:
            # Standard prediction without uncertainty
            predictions = model.predict(X)
            result_df[f"{target_col}_predicted"] = predictions
            
            # If target is available, calculate error
            if target_col in result_df.columns:
                result_df[f"{target_col}_error"] = predictions - result_df[target_col]
                result_df[f"{target_col}_abs_error"] = np.abs(result_df[f"{target_col}_error"])
        
        return result_df
    
    def analyze(
        self,
        model_names: Optional[List[str]] = None,
        data_path: Optional[str] = None
    ) -> None:
        """
        Perform analysis of model results.
        
        Args:
            model_names: Optional list of model names to analyze
            data_path: Optional explicit path to data file
        """
        from flexseq.utils.helpers import progress_bar, ProgressCallback
        
        # Determine which models to analyze
        if model_names is None:
            model_names = list(self.models.keys())
            
            # If no models in memory, use enabled models from config
            if not model_names:
                model_names = get_enabled_models(self.config)
                
        if not model_names:
            logger.warning("No models specified for analysis")
            return
        
        # Load data if needed
        with ProgressCallback(total=1, desc="Loading data") as pbar:
            df = self.load_data(data_path)
            pbar.update()
        
        # Split data
        with ProgressCallback(total=1, desc="Preparing test data") as pbar:
            _, _, test_df = split_data(df, self.config)
            pbar.update()
        
        # Generate predictions for each model
        predictions = {}
        feature_importances = {}
        
        for model_name in progress_bar(model_names, desc="Analyzing models"):
            try:
                # Load model if not in memory
                if model_name in self.models:
                    model = self.models[model_name]
                else:
                    with ProgressCallback(total=1, desc=f"Loading {model_name}", leave=False) as pbar:
                        model = self.load_model(model_name)
                        pbar.update()
                
                # Prepare data
                with ProgressCallback(total=1, desc="Preparing features", leave=False) as pbar:
                    X_test, y_test, feature_names = prepare_data_for_model(test_df, self.config)
                    pbar.update()
                
                # Generate predictions
                with ProgressCallback(total=1, desc=f"Predicting with {model_name}", leave=False) as pbar:
                    predictions[model_name] = model.predict(X_test)
                    pbar.update()
                
                
                # Get feature importances if available
                importance = model.get_feature_importance()
                if importance:
                    feature_importances[model_name] = importance
                    
                    # Save feature importance to CSV
                    importance_df = pd.DataFrame({
                        'feature': list(importance.keys()),
                        'importance': list(importance.values())
                    })
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    
                    output_dir = self.config["paths"]["output_dir"]
                    importance_path = os.path.join(
                        output_dir, 
                        "feature_importance", 
                        f"{model_name}_feature_importance.csv"
                    )
                    os.makedirs(os.path.dirname(importance_path), exist_ok=True)
                    importance_df.to_csv(importance_path, index=False)
                    logger.info(f"Saved feature importance to {importance_path}")
                    
            except Exception as e:
                logger.error(f"Error analyzing model {model_name}: {e}")
        
        # Generate combined results
        target_col = self.config["dataset"]["target"]
        target_values = test_df[target_col].values
        
        # Generate a combined results dataframe
        combined_df = test_df.copy()
        
        for model_name, preds in predictions.items():
            combined_df[f"{model_name}_predicted"] = preds
            combined_df[f"{model_name}_error"] = preds - combined_df[target_col]
            combined_df[f"{model_name}_abs_error"] = np.abs(combined_df[f"{model_name}_error"])
        
        # Save combined results
        output_dir = self.config["paths"]["output_dir"]
        combined_path = os.path.join(output_dir, "combined_analysis_results.csv")
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Saved combined analysis results to {combined_path}")
        
            # Import visualization functions
        from flexseq.utils.visualization import (
            plot_r2_comparison,
            plot_residue_level_rmsf,
            plot_amino_acid_error_analysis,
            plot_amino_acid_error_boxplot,
            plot_amino_acid_scatter_plot,
            plot_error_analysis_by_property,
            plot_r2_comparison_scatter,
            plot_scatter_with_density_contours,
            plot_flexibility_vs_dihedral_angles,
            plot_flexibility_sequence_neighborhood,
            plot_error_response_surface,
            plot_secondary_structure_error_correlation
        )

        # Prepare data for visualization
        predictions = {}
        for model_name in model_names:
            pred_col = f"{model_name}_predicted"
            if pred_col in combined_df.columns:
                predictions[model_name] = combined_df[pred_col].values

        # Generate various visualizations
        plot_r2_comparison(predictions, combined_df[target_col].values, model_names, self.config)
        plot_residue_level_rmsf(combined_df, predictions, target_col, model_names, self.config)
        plot_amino_acid_error_analysis(combined_df, predictions, target_col, model_names, self.config)
        plot_amino_acid_error_boxplot(combined_df, predictions, target_col, model_names, self.config)
        plot_amino_acid_scatter_plot(combined_df, predictions, target_col, model_names, self.config)
        plot_error_analysis_by_property(combined_df, predictions, target_col, model_names, self.config)
        plot_r2_comparison_scatter(predictions, combined_df[target_col].values, model_names, self.config)
        plot_scatter_with_density_contours(combined_df, predictions, target_col, model_names, self.config)
        plot_flexibility_vs_dihedral_angles(combined_df, predictions, target_col, model_names, self.config)
        plot_flexibility_sequence_neighborhood(combined_df, predictions, target_col, model_names, self.config)
        plot_error_response_surface(combined_df, predictions, target_col, model_names, self.config)
        plot_secondary_structure_error_correlation(combined_df, predictions, target_col, model_names, self.config)

        
        # Generate residue-level analysis
        self.residue_level_analysis(combined_df, model_names)
        
        # Generate secondary structure analysis
        self.secondary_structure_analysis(combined_df, model_names)
        
        # Generate amino acid type analysis
        self.amino_acid_analysis(combined_df, model_names)
        
        # Process visualization data (CSV files for later visualization)
        self.generate_visualization_data(combined_df, model_names)
    
    def residue_level_analysis(self, df: pd.DataFrame, model_names: List[str]) -> None:
        """
        Perform residue-level analysis of prediction errors.
        
        Args:
            df: DataFrame with predictions and actual values
            model_names: List of model names that have been analyzed
        """
        target_col = self.config["dataset"]["target"]
        output_dir = self.config["paths"]["output_dir"]
        residue_dir = os.path.join(output_dir, "residue_analysis")
        os.makedirs(residue_dir, exist_ok=True)
        
        # Calculate error statistics per residue position
        residue_stats = []
        
        # Group by normalized residue position (if available) or by residue ID
        groupby_col = "normalized_resid" if "normalized_resid" in df.columns else "resid"
        
        # Bin values if using normalized_resid
        if groupby_col == "normalized_resid":
            df["resid_bin"] = pd.cut(df[groupby_col], bins=20, labels=False)
            groupby_col = "resid_bin"
        
        for pos, group in df.groupby(groupby_col):
            row = {groupby_col: pos, "count": len(group)}
            
            for model_name in model_names:
                error_col = f"{model_name}_abs_error"
                if error_col in group.columns:
                    row[f"{model_name}_mean_error"] = group[error_col].mean()
                    row[f"{model_name}_median_error"] = group[error_col].median()
                    row[f"{model_name}_std_error"] = group[error_col].std()
            
            residue_stats.append(row)
        
        if residue_stats:
            residue_df = pd.DataFrame(residue_stats)
            residue_path = os.path.join(residue_dir, "residue_position_errors.csv")
            residue_df.to_csv(residue_path, index=False)
            logger.info(f"Saved residue position analysis to {residue_path}")
    
    def secondary_structure_analysis(self, df: pd.DataFrame, model_names: List[str]) -> None:
        """
        Perform secondary structure analysis of prediction errors.
        
        Args:
            df: DataFrame with predictions and actual values
            model_names: List of model names that have been analyzed
        """
        if "secondary_structure_encoded" not in df.columns:
            logger.warning("Secondary structure information not available for analysis")
            return
        
        output_dir = self.config["paths"]["output_dir"]
        ss_dir = os.path.join(output_dir, "residue_analysis")
        os.makedirs(ss_dir, exist_ok=True)
        
        # Map encoded values to types
        ss_mapping = {0: "helix", 1: "sheet", 2: "loop"}
        
        # Calculate error statistics per secondary structure type
        ss_stats = []
        
        for ss_code, group in df.groupby("secondary_structure_encoded"):
            ss_type = ss_mapping.get(ss_code, f"unknown_{ss_code}")
            row = {"secondary_structure": ss_type, "count": len(group)}
            
            for model_name in model_names:
                error_col = f"{model_name}_abs_error"
                if error_col in group.columns:
                    row[f"{model_name}_mean_error"] = group[error_col].mean()
                    row[f"{model_name}_median_error"] = group[error_col].median()
                    row[f"{model_name}_std_error"] = group[error_col].std()
            
            ss_stats.append(row)
        
        if ss_stats:
            ss_df = pd.DataFrame(ss_stats)
            ss_path = os.path.join(ss_dir, "secondary_structure_errors.csv")
            ss_df.to_csv(ss_path, index=False)
            logger.info(f"Saved secondary structure analysis to {ss_path}")
    
    def amino_acid_analysis(self, df: pd.DataFrame, model_names: List[str]) -> None:
        """
        Perform amino acid-specific analysis of prediction errors.
        
        Args:
            df: DataFrame with predictions and actual values
            model_names: List of model names that have been analyzed
        """
        output_dir = self.config["paths"]["output_dir"]
        aa_dir = os.path.join(output_dir, "residue_analysis")
        os.makedirs(aa_dir, exist_ok=True)
        
        # Calculate error statistics per amino acid type
        aa_stats = []
        
        for aa, group in df.groupby("resname"):
            row = {"resname": aa, "count": len(group)}
            
            for model_name in model_names:
                error_col = f"{model_name}_abs_error"
                if error_col in group.columns:
                    row[f"{model_name}_mean_error"] = group[error_col].mean()
                    row[f"{model_name}_median_error"] = group[error_col].median()
                    row[f"{model_name}_std_error"] = group[error_col].std()
            
            aa_stats.append(row)
        
        if aa_stats:
            aa_df = pd.DataFrame(aa_stats)
            aa_path = os.path.join(aa_dir, "amino_acid_errors.csv")
            aa_df.to_csv(aa_path, index=False)
            logger.info(f"Saved amino acid analysis to {aa_path}")
    
    def generate_visualization_data(self, df: pd.DataFrame, model_names: List[str]) -> None:
        """
        Generate data files for visualizations.
        
        Args:
            df: DataFrame with predictions and actual values
            model_names: List of model names that have been analyzed
        """
        target_col = self.config["dataset"]["target"]
        output_dir = self.config["paths"]["output_dir"]
        vis_dir = os.path.join(output_dir, "visualization_data")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate histogram data for RMSF distributions
        try:
            histogram_data = []
            
            # Get actual values
            actual_values = df[target_col].dropna()
            
            actual_hist, actual_bins = np.histogram(actual_values, bins=20)
            
            for i in range(len(actual_hist)):
                histogram_data.append({
                    'source': 'actual',
                    'bin_start': actual_bins[i],
                    'bin_end': actual_bins[i+1],
                    'count': actual_hist[i]
                })
            
            # Get predicted values for each model
            for model_name in model_names:
                pred_col = f"{model_name}_predicted"
                if pred_col in df.columns:
                    pred_values = df[pred_col].dropna()
                    pred_hist, pred_bins = np.histogram(pred_values, bins=actual_bins)
                    
                    for i in range(len(pred_hist)):
                        histogram_data.append({
                            'source': model_name,
                            'bin_start': pred_bins[i],
                            'bin_end': pred_bins[i+1],
                            'count': pred_hist[i]
                        })
            
            # Save histogram data
            histogram_df = pd.DataFrame(histogram_data)
            histogram_path = os.path.join(vis_dir, "rmsf_distribution.csv")
            histogram_df.to_csv(histogram_path, index=False)
            logger.info(f"Saved RMSF distribution data to {histogram_path}")
            
        except Exception as e:
            logger.error(f"Error generating histogram data: {e}")
        
        # Generate scatter plot data
        try:
            scatter_data = []
            
            # Sample rows to avoid too large data files
            sample_size = min(5000, len(df))
            sampled_df = df.sample(sample_size, random_state=self.config["system"]["random_state"])
            
            for _, row in sampled_df.iterrows():
                data_point = {
                    'domain_id': row['domain_id'],
                    'resid': row['resid'],
                    'resname': row['resname'],
                    'actual': row[target_col]
                }
                
                # Add predicted values
                for model_name in model_names:
                    pred_col = f"{model_name}_predicted"
                    if pred_col in row:
                        data_point[model_name] = row[pred_col]
                
                # Add structural features if available
                for feature in ['secondary_structure_encoded', 'core_exterior_encoded', 'normalized_resid']:
                    if feature in row:
                        data_point[feature] = row[feature]
                
                scatter_data.append(data_point)
            
            # Save scatter data
            scatter_df = pd.DataFrame(scatter_data)
            scatter_path = os.path.join(vis_dir, "rmsf_scatter_data.csv")
            scatter_df.to_csv(scatter_path, index=False)
            logger.info(f"Saved scatter plot data to {scatter_path}")
            
        except Exception as e:
            logger.error(f"Error generating scatter plot data: {e}")
    
    def run_pipeline(
        self, 
        model_names: Optional[List[str]] = None,
        data_path: Optional[str] = None,
        skip_visualization: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Run the complete pipeline: train, evaluate, and analyze.
        
        Args:
            model_names: Optional list of model names to use
            data_path: Optional explicit path to data file
            skip_visualization: Whether to skip visualization data generation
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        # Train models
        self.train(model_names, data_path)
        
        # Evaluate models
        results = self.evaluate(model_names, data_path)
        
        # Analyze (optional)
        if not skip_visualization:
            self.analyze(model_names, data_path)
        
        return results