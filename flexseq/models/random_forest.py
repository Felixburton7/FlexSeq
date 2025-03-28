"""
Random Forest model implementation for the FlexSeq ML pipeline.

This module provides a RandomForestModel for protein flexibility prediction
with support for uncertainty estimation and hyperparameter optimization.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from flexseq.models import register_model
from flexseq.models.base import BaseModel
from flexseq.utils.helpers import ProgressCallback

logger = logging.getLogger(__name__)

@register_model("random_forest")
class RandomForestModel(BaseModel):
    """
    Random Forest model for protein flexibility prediction.
    
    This model uses an ensemble of decision trees to capture non-linear
    relationships between protein features and flexibility.
    """
    
    def __init__(
        self, 
        n_estimators: int = 100, 
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, float, int] = 0.7,
        bootstrap: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree (None = unlimited)
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters passed to RandomForestRegressor
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.model_params = kwargs
        self.model = None
        self.feature_names_ = None
        self.best_params_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], feature_names: Optional[List[str]] = None) -> 'RandomForestModel':
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target RMSF values
            feature_names: Optional list of feature names
            
        Returns:
            Self, for method chaining
        """
        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        elif feature_names is not None:
            self.feature_names_ = feature_names
        
        # Check if randomized search is enabled in the model parameters
        use_randomized_search = self.model_params.pop('use_randomized_search', False)
        
        if use_randomized_search:
            # Get hyperparameter search config
            n_iter = self.model_params.pop('n_iter', 20)
            cv = self.model_params.pop('cv', 3)
            param_distributions = self.model_params.pop('param_distributions', None)
            
            # Use default param distributions if not provided
            if param_distributions is None:
                param_distributions = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2', 0.7],
                    'bootstrap': [True, False]
                }
            
            # Base estimator with fixed random_state
            base_rf = RandomForestRegressor(random_state=self.random_state, **self.model_params)
            
            with ProgressCallback(total=1, desc="Setting up RandomizedSearchCV") as pbar:
                logger.info(f"Setting up RandomizedSearchCV with {n_iter} iterations and {cv} folds")
                
                # Create the randomized search
                search = RandomizedSearchCV(
                    base_rf,
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=0,
                    return_train_score=True
                )
                pbar.update()
            
            # Fit the randomized search
            with ProgressCallback(total=1, desc="Training with RandomizedSearchCV") as pbar:
                search.fit(X, y)
                self.model = search.best_estimator_
                self.best_params_ = search.best_params_
                pbar.update()
                
            logger.info(f"Best hyperparameters: {self.best_params_}")
        else:
            # Create and train a standard RandomForestRegressor
            with ProgressCallback(total=1, desc="Training Random Forest") as pbar:
                self.model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    bootstrap=self.bootstrap,
                    random_state=self.random_state,
                    n_jobs=-1,
                    **self.model_params
                )
                
                self.model.fit(X, y)
                pbar.update()
        
        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate RMSF predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted RMSF values
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_with_std(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate RMSF predictions with standard deviation (uncertainty).
        
        Uses the variance of predictions across the ensemble of trees
        as a measure of prediction uncertainty.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, std_dev) arrays
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        # Make predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Calculate mean and standard deviation
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return mean_prediction, std_prediction
    
    def hyperparameter_optimize(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        param_grid: Dict[str, Any],
        method: str = "bayesian",
        n_trials: int = 20,
        cv: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid or distributions
            method: Optimization method ("grid", "random", or "bayesian")
            n_trials: Number of trials for random or bayesian methods
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters
        """
        # Random Forest ignores the method and n_trials parameters, using RandomizedSearchCV instead
        if method != "random":
            logger.warning(f"RandomForest only supports 'random' method for optimization, ignoring '{method}'")
            
        with ProgressCallback(total=1, desc="Hyperparameter optimization") as pbar:
            search = RandomizedSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                param_distributions=param_grid,
                n_iter=n_trials,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
                return_train_score=True
            )
            
            search.fit(X, y)
            pbar.update()
            
        # Update model with the best estimator
        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        
        logger.info(f"Best hyperparameters: {self.best_params_}")
        
        return self.best_params_
        
    def save(self, path: str) -> None:
        """
        Save model to disk using joblib.
        
        Args:
            path: Path to save location
        """
        if self.model is None:
            raise RuntimeError("Cannot save untrained model")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model state
        state = {
            'model': self.model,
            'feature_names': self.feature_names_,
            'best_params': self.best_params_,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'random_state': self.random_state,
                'model_params': self.model_params
            }
        }
        
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'RandomForestModel':
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded RandomForestModel instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            state = joblib.load(path)
            
            # Create new instance with saved parameters
            params = state['params']
            instance = cls(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                max_features=params.get('max_features', 0.7),
                bootstrap=params.get('bootstrap', True),
                random_state=params.get('random_state', 42),
                **params.get('model_params', {})
            )
            
            # Restore model and feature names
            instance.model = state['model']
            instance.feature_names_ = state.get('feature_names', None)
            instance.best_params_ = state.get('best_params', None)
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
    def get_feature_importance(self, X_val=None, y_val=None) -> Dict[str, float]:
        """
        Get feature importance values using permutation importance.
        
        Args:
            X_val: Optional validation features for permutation importance
            y_val: Optional validation targets for permutation importance
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if self.model is None:
            return {}
        
        # If validation data is provided, use permutation importance
        if X_val is not None and y_val is not None and len(X_val) > 0:
            try:
                from sklearn.inspection import permutation_importance
                
                # Calculate permutation importance
                r = permutation_importance(
                    self.model, X_val, y_val, 
                    n_repeats=10, 
                    random_state=self.random_state
                )
                
                # Use mean importance as the feature importance
                importance_values = r.importances_mean
                
                # Map to feature names if available
                if self.feature_names_ is not None and len(self.feature_names_) == len(importance_values):
                    return dict(zip(self.feature_names_, importance_values))
                else:
                    return {f"feature_{i}": imp for i, imp in enumerate(importance_values)}
                    
            except Exception as e:
                logger.warning(f"Could not compute permutation importance: {e}")
                # Fall back to built-in feature importance
        
        # Use built-in feature importance as fallback
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_
            
            if self.feature_names_ is not None and len(self.feature_names_) == len(importance_values):
                return dict(zip(self.feature_names_, importance_values))
            else:
                return {f"feature_{i}": importance for i, importance in enumerate(importance_values)}
        
        return {}