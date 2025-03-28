"""
Neural Network model implementation for the FlexSeq ML pipeline.

This module provides a PyTorch-based neural network model
for protein flexibility prediction, with support for
hyperparameter optimization.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from flexseq.models import register_model
from flexseq.models.base import BaseModel
from flexseq.utils.helpers import ProgressCallback, progress_bar

logger = logging.getLogger(__name__)

class FlexibilityNN(nn.Module):
    """
    Neural network architecture for protein flexibility prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [64, 32],
        activation: str = "relu",
        dropout: float = 0.2
    ):
        """
        Initialize the neural network architecture.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            activation: Activation function to use
            dropout: Dropout rate
        """
        super(FlexibilityNN, self).__init__()
        
        # Define activation function
        if activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "leaky_relu":
            act_fn = nn.LeakyReLU()
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh()
        elif activation.lower() == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            act_fn = nn.ReLU()
            logger.warning(f"Unknown activation function '{activation}', using ReLU")
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output layer (single value for RMSF prediction)
        layers.append(nn.Linear(prev_dim, 1))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x).squeeze()

@register_model("neural_network")
class NeuralNetworkModel(BaseModel):
    """
    Neural Network model for protein flexibility prediction.
    
    This model uses a feed-forward neural network to learn complex
    relationships between protein features and flexibility.
    """
    
    def __init__(
        self,
        architecture: Dict[str, Any] = None,
        training: Dict[str, Any] = None,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the Neural Network model.
        
        Args:
            architecture: Dictionary of architecture parameters
            training: Dictionary of training parameters
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        # Set default architecture if not provided
        if architecture is None:
            architecture = {
                "hidden_layers": [64, 32],
                "activation": "relu",
                "dropout": 0.2
            }
        
        # Set default training parameters if not provided
        if training is None:
            training = {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "early_stopping": True,
                "patience": 10
            }
        
        self.architecture = architecture
        self.training = training
        self.random_state = random_state
        self.model = None
        self.feature_names_ = None
        self.scaler = None
        self.history = None
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def _create_model(self, input_dim: int) -> FlexibilityNN:
        """
        Create the neural network model.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Initialized FlexibilityNN model
        """
        model = FlexibilityNN(
            input_dim=input_dim,
            hidden_layers=self.architecture.get("hidden_layers", [64, 32]),
            activation=self.architecture.get("activation", "relu"),
            dropout=self.architecture.get("dropout", 0.2)
        )
        return model.to(self.device)
    
    def _get_optimizer(self, model: FlexibilityNN) -> optim.Optimizer:
        """
        Get the appropriate optimizer.
        
        Args:
            model: The neural network model
            
        Returns:
            Configured optimizer
        """
        optimizer_name = self.training.get("optimizer", "adam").lower()
        lr = self.training.get("learning_rate", 0.001)
        
        if optimizer_name == "adam":
            return optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            return optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=lr)
        else:
            logger.warning(f"Unknown optimizer '{optimizer_name}', using Adam")
            return optim.Adam(model.parameters(), lr=lr)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], feature_names: Optional[List[str]] = None) -> 'NeuralNetworkModel':
        """
        Train the Neural Network model.
        
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
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Convert target to numpy array if needed
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Scale features using StandardScaler
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Create PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_array)
        
        # Create validation split (20% of training data)
        from sklearn.model_selection import train_test_split
        train_indices, val_indices = train_test_split(
            np.arange(len(X_scaled)), 
            test_size=0.2, 
            random_state=self.random_state
        )
        
        # Create dataset and dataloader for training
        train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
        batch_size = self.training.get("batch_size", 32)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X_array.shape[1]
        self.model = self._create_model(input_dim)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = self._get_optimizer(self.model)
        
        # Training parameters
        epochs = self.training.get("epochs", 100)
        early_stopping = self.training.get("early_stopping", True)
        patience = self.training.get("patience", 10)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rate': []
        }
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        # Use progress bar for epochs
        epoch_pbar = progress_bar(range(epochs), desc="Training NN")
        
        for epoch in epoch_pbar:
            # Training phase
            self.model.train()
            train_running_loss = 0.0
            train_preds = []
            train_targets = []
            
            # Track batch progress
            batch_pbar = progress_bar(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch_X, batch_y in batch_pbar:
                # Move tensors to the configured device
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_running_loss += loss.item() * batch_X.size(0)
                
                # Collect predictions and targets for R² calculation
                train_preds.append(outputs.detach().cpu().numpy())
                train_targets.append(batch_y.detach().cpu().numpy())
                
                # Update batch progress bar
                batch_pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Compute average epoch loss for training
            train_epoch_loss = train_running_loss / len(train_dataset)
            
            # Concatenate predictions and targets for full training set R²
            train_preds = np.concatenate(train_preds)
            train_targets = np.concatenate(train_targets)
            
            # Calculate R² for training set
            from sklearn.metrics import r2_score
            train_r2 = r2_score(train_targets, train_preds)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_X = X_tensor[val_indices].to(self.device)
                val_y = y_tensor[val_indices].to(self.device)
                val_outputs = self.model(val_X)
                val_batch_loss = criterion(val_outputs, val_y)
                val_loss = val_batch_loss.item()
                
                # Calculate R² for validation set
                val_preds = val_outputs.cpu().numpy()
                val_targets = val_y.cpu().numpy()
                val_r2 = r2_score(val_targets, val_preds)
            
            # Store metrics in history
            self.history['train_loss'].append(train_epoch_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_r2'].append(train_r2)
            self.history['val_r2'].append(val_r2)
            self.history['learning_rate'].append(self.training.get("learning_rate", 0.001))
            
            # Update epoch progress bar
            epoch_pbar.set_postfix(
                train_loss=f"{train_epoch_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_r2=f"{val_r2:.4f}"
            )
            
            # Early stopping check
            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_state = {
                        'model_state': self.model.state_dict(),
                        'input_dim': input_dim
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        # Restore best model state
                        self.model.load_state_dict(best_state['model_state'])
                        break
        
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
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def predict_with_std(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates using MC Dropout.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, std_deviation)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Set model to training mode to enable dropout for MC Dropout
        self.model.train()
        
        # Perform multiple forward passes with dropout enabled
        n_samples = 30  # Number of MC samples
        samples = []
        
        with torch.no_grad():  # No gradients needed
            for _ in range(n_samples):
                predictions = self.model(X_tensor).cpu().numpy()
                samples.append(predictions)
        
        # Calculate mean and standard deviation across samples
        samples = np.stack(samples, axis=0)
        mean_prediction = np.mean(samples, axis=0)
        std_prediction = np.std(samples, axis=0)
        
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
        if method.lower() == "bayesian":
            # Try to use optuna for Bayesian optimization
            try:
                import optuna
                logger.info("Using Optuna for Bayesian hyperparameter optimization")
                return self._bayesian_optimization(X, y, param_grid, n_trials, cv)
            except ImportError:
                logger.warning("Optuna not available, falling back to random search")
                method = "random"
        
        if method.lower() == "random":
            return self._random_optimization(X, y, param_grid, n_trials, cv)
        
        elif method.lower() == "grid":
            return self._grid_optimization(X, y, param_grid, cv)
        
        else:
            logger.warning(f"Unknown optimization method '{method}', using random search")
            return self._random_optimization(X, y, param_grid, n_trials, cv)
    
    def _random_optimization(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        param_grid: Dict[str, Any],
        n_trials: int,
        cv: int
    ) -> Dict[str, Any]:
        """
        Perform random search hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid
            n_trials: Number of trials
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Create KFold cross-validator
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Create parameter combinations
        from itertools import product
        import random
        
        # Get all possible combinations
        param_combinations = []
        
        # If hidden_layers is in the grid
        if 'hidden_layers' in param_grid:
            hidden_layer_options = param_grid['hidden_layers']
            rest_params = {k: v for k, v in param_grid.items() if k != 'hidden_layers'}
        else:
            hidden_layer_options = [self.architecture.get('hidden_layers', [64, 32])]
            rest_params = param_grid
            
        # Get combinations of the rest
        keys = list(rest_params.keys())
        values = list(rest_params.values())
        
        for hidden_layer in hidden_layer_options:
            for combination in product(*values):
                param_combo = dict(zip(keys, combination))
                param_combo['hidden_layers'] = hidden_layer
                param_combinations.append(param_combo)
                
        # Limit to n_trials
        if len(param_combinations) > n_trials:
            random.seed(self.random_state)
            param_combinations = random.sample(param_combinations, n_trials)
            
        logger.info(f"Performing random search with {len(param_combinations)} parameter combinations")
        
        # Train and evaluate each combination
        results = []
        
        for i, params in enumerate(progress_bar(param_combinations, desc="Parameter combinations")):
            # Extract architecture and training params
            arch_params = {
                'hidden_layers': params.get('hidden_layers', self.architecture.get('hidden_layers', [64, 32])),
                'activation': params.get('activation', self.architecture.get('activation', 'relu')),
                'dropout': params.get('dropout', self.architecture.get('dropout', 0.2))
            }
            
            train_params = {
                'optimizer': params.get('optimizer', self.training.get('optimizer', 'adam')),
                'learning_rate': params.get('learning_rate', self.training.get('learning_rate', 0.001)),
                'batch_size': params.get('batch_size', self.training.get('batch_size', 32)),
                'epochs': params.get('epochs', self.training.get('epochs', 100)),
                'early_stopping': params.get('early_stopping', self.training.get('early_stopping', True)),
                'patience': params.get('patience', self.training.get('patience', 10))
            }
            
            # Perform cross-validation
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X_array):
                X_train, X_val = X_array[train_idx], X_array[val_idx]
                y_train, y_val = y_array[train_idx], y_array[val_idx]
                
                # Create and train model
                model = NeuralNetworkModel(
                    architecture=arch_params,
                    training=train_params,
                    random_state=self.random_state
                )
                
                # Limit epochs for CV
                model.training['epochs'] = min(model.training['epochs'], 30)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                preds = model.predict(X_val)
                from sklearn.metrics import mean_squared_error
                score = -mean_squared_error(y_val, preds)  # Negative MSE (higher is better)
                cv_scores.append(score)
            
            # Store results
            mean_score = np.mean(cv_scores)
            results.append((mean_score, arch_params, train_params))
            logger.debug(f"Parameters: {params}, Score: {mean_score:.4f}")
        
        # Find best parameters
        results.sort(reverse=True)  # Higher score is better
        best_score, best_arch, best_train = results[0]
        
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best architecture parameters: {best_arch}")
        logger.info(f"Best training parameters: {best_train}")
        
        # Update model parameters
        self.architecture = best_arch
        self.training = best_train
        
        # Train final model on all data
        self.fit(X, y)
        
        # Return combined parameters
        return {**best_arch, **best_train}
    
    def _grid_optimization(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        param_grid: Dict[str, Any],
        cv: int
    ) -> Dict[str, Any]:
        """
        Perform grid search hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters
        """
        # Grid search is same as random search but with all combinations
        from itertools import product
        
        # Get all possible combinations
        param_combinations = []
        
        # If hidden_layers is in the grid
        if 'hidden_layers' in param_grid:
            hidden_layer_options = param_grid['hidden_layers']
            rest_params = {k: v for k, v in param_grid.items() if k != 'hidden_layers'}
        else:
            hidden_layer_options = [self.architecture.get('hidden_layers', [64, 32])]
            rest_params = param_grid
            
        # Get combinations of the rest
        keys = list(rest_params.keys())
        values = list(rest_params.values())
        
        for hidden_layer in hidden_layer_options:
            for combination in product(*values):
                param_combo = dict(zip(keys, combination))
                param_combo['hidden_layers'] = hidden_layer
                param_combinations.append(param_combo)
                
        logger.info(f"Performing grid search with {len(param_combinations)} parameter combinations")
        
        # Use random optimization with all combinations
        n_trials = len(param_combinations)
        return self._random_optimization(X, y, param_grid, n_trials, cv)
    
    def _bayesian_optimization(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        param_grid: Dict[str, Any],
        n_trials: int,
        cv: int
    ) -> Dict[str, Any]:
        """
        Perform Bayesian hyperparameter optimization using Optuna.
        
        Args:
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid
            n_trials: Number of trials
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters
        """
        try:
            import optuna
        except ImportError:
            logger.warning("Optuna not available, falling back to random search")
            return self._random_optimization(X, y, param_grid, n_trials, cv)
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Create KFold cross-validator
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Define objective function for Optuna
        def objective(trial):
            # Suggest hyperparameters
            hidden_layers_options = param_grid.get('hidden_layers', [[64, 32], [128, 64], [32, 16]])
            architecture = {
                'hidden_layers': trial.suggest_categorical('hidden_layers', hidden_layers_options),
                'activation': trial.suggest_categorical('activation', param_grid.get('activation', ['relu', 'leaky_relu'])),
                'dropout': trial.suggest_float('dropout', 
                                            low=min(param_grid.get('dropout', [0.1, 0.5])),
                                            high=max(param_grid.get('dropout', [0.1, 0.5])))
            }
            
            learning_rates = param_grid.get('learning_rate', [0.0001, 0.01])
            batch_sizes = param_grid.get('batch_size', [16, 32, 64])
            patience_range = param_grid.get('patience', [5, 15])
            epochs_range = param_grid.get('epochs', [50, 100])
            optimizer_options = param_grid.get('optimizer', ['adam', 'rmsprop'])
            
            training = {
                'optimizer': trial.suggest_categorical('optimizer', optimizer_options),
                'learning_rate': trial.suggest_float('learning_rate', 
                                                low=min(learning_rates),
                                                high=max(learning_rates),
                                                log=True),
                'batch_size': trial.suggest_categorical('batch_size', batch_sizes),
                'early_stopping': True,
                'patience': trial.suggest_int('patience', 
                                            low=min(patience_range),
                                            high=max(patience_range)),
                'epochs': trial.suggest_int('epochs', 
                                        low=min(epochs_range),
                                        high=max(epochs_range))
            }
            
            # Perform cross-validation
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X_array):
                X_train, X_val = X_array[train_idx], X_array[val_idx]
                y_train, y_val = y_array[train_idx], y_array[val_idx]
                
                # Create and train model
                model = NeuralNetworkModel(
                    architecture=architecture,
                    training=training,
                    random_state=self.random_state
                )
                
                # Limit epochs for CV
                model.training['epochs'] = min(model.training['epochs'], 30)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                preds = model.predict(X_val)
                from sklearn.metrics import mean_squared_error
                score = -mean_squared_error(y_val, preds)  # Negative MSE (higher is better)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        # Create and run Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Construct best parameters dictionary
        best_hidden_layers = param_grid.get('hidden_layers', [[64, 32], [128, 64], [32, 16]])
        hidden_layers_idx = 0
        if 'hidden_layers' in best_params:
            for i, layers in enumerate(param_grid.get('hidden_layers', [])):
                if str(layers) == str(best_params['hidden_layers']):
                    hidden_layers_idx = i
                    break
        
        best_arch = {
            'hidden_layers': param_grid.get('hidden_layers', [[64, 32]])[hidden_layers_idx] if param_grid.get('hidden_layers') else [64, 32],
            'activation': best_params.get('activation', self.architecture.get('activation', 'relu')),
            'dropout': best_params.get('dropout', self.architecture.get('dropout', 0.2))
        }
        
        best_train = {
            'optimizer': best_params.get('optimizer', self.training.get('optimizer', 'adam')),
            'learning_rate': best_params.get('learning_rate', self.training.get('learning_rate', 0.001)),
            'batch_size': best_params.get('batch_size', self.training.get('batch_size', 32)),
            'early_stopping': True,
            'patience': best_params.get('patience', self.training.get('patience', 10)),
            'epochs': best_params.get('epochs', self.training.get('epochs', 100))
        }
        
        # Update model parameters
        self.architecture = best_arch
        self.training = best_train
        
        # Train final model on all data
        self.fit(X, y)
        
        # Return combined parameters
        return {**best_arch, **best_train}
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save location
        """
        if self.model is None:
            raise RuntimeError("Cannot save untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model state
        state = {
            'model_state': self.model.state_dict(),
            'architecture': self.architecture,
            'training': self.training,
            'random_state': self.random_state,
            'feature_names': self.feature_names_,
            'scaler': self.scaler,
            'history': self.history,
            'input_dim': self.model.model[0].in_features  # Get input dimension from first layer
        }
        
        # Save state
        torch.save(state, path)
        logger.info(f"Model saved to {path}")
        
        # Save training history separately as CSV
        if self.history:
            history_path = os.path.splitext(path)[0] + "_history.csv"
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(history_path, index=False)
            logger.info(f"Training history saved to {history_path}")
    
    @classmethod
    def load(cls, path: str) -> 'NeuralNetworkModel':
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded NeuralNetworkModel instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            # Load state dictionary
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            state = torch.load(path, map_location=device)
            
            # Create new instance with saved parameters
            instance = cls(
                architecture=state['architecture'],
                training=state['training'],
                random_state=state['random_state']
            )
            
            # Restore feature names, scaler, and history
            instance.feature_names_ = state['feature_names']
            instance.scaler = state['scaler']
            instance.history = state.get('history')
            
            # Create and restore model
            input_dim = state['input_dim']
            instance.model = instance._create_model(input_dim)
            instance.model.load_state_dict(state['model_state'])
            instance.model.eval()
            
            # Load training history if available
            history_path = os.path.splitext(path)[0] + "_history.csv"
            if os.path.exists(history_path):
                try:
                    history_df = pd.read_csv(history_path)
                    instance.history = {col: history_df[col].tolist() for col in history_df.columns}
                except Exception as e:
                    logger.warning(f"Could not load training history: {e}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self, X_val=None, y_val=None) -> Optional[Dict[str, float]]:
        """
        Get feature importance values using permutation importance.
        
        Args:
            X_val: Optional validation features for permutation importance
            y_val: Optional validation targets for permutation importance
            
        Returns:
            Dictionary mapping feature names to importance values or None
        """
        if self.model is None:
            return None
        
        # If validation data is provided, use permutation importance
        if X_val is not None and y_val is not None and len(X_val) > 0:
            try:
                from sklearn.inspection import permutation_importance
                
                # Set model to evaluation mode
                self.model.eval()
                
                # Define a prediction function for permutation importance
                def predict_fn(X_test):
                    X_tensor = torch.FloatTensor(self.scaler.transform(X_test)).to(self.device)
                    with torch.no_grad():
                        return self.model(X_tensor).cpu().numpy()
                
                # Calculate permutation importance
                r = permutation_importance(
                    predict_fn, X_val, y_val, 
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
                # Fall back to weight-based importance
        
        # Fall back to weight-based importance
        try:
            # Get weights from the first layer
            first_layer = self.model.model[0]
            weights = first_layer.weight.data.cpu().numpy()
            
            # Use absolute values of weights as importance
            importance = np.mean(np.abs(weights), axis=0)
            
            # Map to feature names if available
            if self.feature_names_ is not None and len(self.feature_names_) == len(importance):
                return dict(zip(self.feature_names_, importance))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importance)}
                
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")
            return None
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """
        Get training history if available.
        
        Returns:
            Dictionary with training metrics by epoch, or None if not available
        """
        return self.history