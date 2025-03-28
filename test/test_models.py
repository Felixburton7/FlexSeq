"""
Test model implementations for the FlexSeq pipeline.

This script tests the Neural Network and Random Forest model implementations:
- Model initialization
- Training and prediction
- Feature importance calculation
- Model saving and loading
- Uncertainty quantification
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd

# Add parent directory to path to import flexseq
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexseq.models.random_forest import RandomForestModel
from flexseq.models.neural_network import NeuralNetworkModel
from flexseq.config import load_config
from flexseq.data.processor import load_and_process_data, prepare_data_for_model

class TestModels(unittest.TestCase):
    """Test the FlexSeq model implementations."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for output and models
        self.temp_dir = tempfile.mkdtemp()
        
        # Base path for test data
        self.test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        self.data_path_320 = os.path.join(self.test_data_dir, 'temperature_320_train.csv')
        
        # Load configuration
        self.config = load_config()
        self.config['temperature']['current'] = 320
        
        # Load and process test data
        self.df = load_and_process_data(self.data_path_320, self.config)
        
        # Prepare data for models
        self.X, self.y, self.feature_names = prepare_data_for_model(self.df, self.config)
        
        # Split into small train/test sets for quick testing
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_random_forest_initialization(self):
        """Test Random Forest model initialization."""
        # Test default initialization
        model = RandomForestModel()
        self.assertIsNotNone(model)
        
        # Test initialization with custom parameters
        model = RandomForestModel(
            n_estimators=20,
            max_depth=5,
            min_samples_split=3,
            random_state=42
        )
        self.assertEqual(model.n_estimators, 20)
        self.assertEqual(model.max_depth, 5)
        self.assertEqual(model.min_samples_split, 3)
    
    def test_neural_network_initialization(self):
        """Test Neural Network model initialization."""
        # Test default initialization
        model = NeuralNetworkModel()
        self.assertIsNotNone(model)
        
        # Test initialization with custom parameters
        model = NeuralNetworkModel(
            architecture={
                "hidden_layers": [32, 16],
                "activation": "relu",
                "dropout": 0.3
            },
            training={
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs": 10
            },
            random_state=42
        )
        self.assertEqual(model.architecture["hidden_layers"], [32, 16])
        self.assertEqual(model.training["batch_size"], 16)
    
    def test_random_forest_train_predict(self):
        """Test Random Forest training and prediction."""
        # Initialize model with minimal estimators for speed
        model = RandomForestModel(n_estimators=5, random_state=42)
        
        # Train
        model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(model.model)
        
        # Predict
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y_test, predictions)
        self.assertGreaterEqual(r2, -1.0)  # Ensure R² is at least -1 (can be poor on small test set)
    
    def test_neural_network_train_predict(self):
        """Test Neural Network training and prediction."""
        # Initialize model with minimal training for speed
        model = NeuralNetworkModel(
            architecture={"hidden_layers": [16, 8]},
            training={"epochs": 3, "batch_size": 16},
            random_state=42
        )
        
        # Train
        model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(model.model)
        
        # Check that history is tracked
        self.assertIsNotNone(model.history)
        self.assertIn("train_loss", model.history)
        self.assertIn("val_loss", model.history)
        
        # Predict
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_random_forest_feature_importance(self):
        """Test Random Forest feature importance calculation."""
        # Initialize and train model
        model = RandomForestModel(n_estimators=5, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Calculate feature importance
        importance = model.get_feature_importance()
        
        # Verify results
        self.assertIsNotNone(importance)
        self.assertGreater(len(importance), 0)
        
        # Sum of importances should be close to 1.0
        self.assertAlmostEqual(sum(importance.values()), 1.0, places=5)
    
    def test_neural_network_feature_importance(self):
        """Test Neural Network feature importance approximation."""
        # Initialize and train model
        model = NeuralNetworkModel(
            architecture={"hidden_layers": [16, 8]},
            training={"epochs": 3, "batch_size": 16},
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        # Calculate feature importance
        importance = model.get_feature_importance()
        
        # Verify results
        self.assertIsNotNone(importance)
        self.assertGreater(len(importance), 0)
    
    def test_random_forest_save_load(self):
        """Test Random Forest model saving and loading."""
        # Initialize and train model
        model = RandomForestModel(n_estimators=5, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Save model
        model_path = os.path.join(self.temp_dir, "rf_model.pkl")
        model.save(model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        loaded_model = RandomForestModel.load(model_path)
        self.assertIsNotNone(loaded_model.model)
        
        # Test predictions from loaded model
        original_preds = model.predict(self.X_test)
        loaded_preds = loaded_model.predict(self.X_test)
        
        # Predictions should be identical
        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)
    
    def test_neural_network_save_load(self):
        """Test Neural Network model saving and loading."""
        # Initialize and train model
        model = NeuralNetworkModel(
            architecture={"hidden_layers": [16, 8]},
            training={"epochs": 3, "batch_size": 16},
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        # Save model
        model_path = os.path.join(self.temp_dir, "nn_model.pkl")
        model.save(model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # Check for history file
        history_path = os.path.splitext(model_path)[0] + "_history.csv"
        self.assertTrue(os.path.exists(history_path))
        
        # Load model
        loaded_model = NeuralNetworkModel.load(model_path)
        self.assertIsNotNone(loaded_model.model)
        
        # Test predictions from loaded model
        original_preds = model.predict(self.X_test)
        loaded_preds = loaded_model.predict(self.X_test)
        
        # Predictions should be nearly identical (may have minor numerical differences)
        np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)
    
    def test_random_forest_uncertainty(self):
        """Test Random Forest uncertainty estimation."""
        # Initialize and train model
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Generate predictions with uncertainty
        predictions, uncertainties = model.predict_with_std(self.X_test)
        
        # Verify results
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(len(uncertainties), len(self.X_test))
        
        # Uncertainties should be non-negative
        self.assertTrue(np.all(uncertainties >= 0))
    
    def test_neural_network_uncertainty(self):
        """Test Neural Network uncertainty estimation (MC Dropout)."""
        # Initialize and train model
        model = NeuralNetworkModel(
            architecture={"hidden_layers": [16, 8], "dropout": 0.3},
            training={"epochs": 3, "batch_size": 16},
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        # Generate predictions with uncertainty
        predictions, uncertainties = model.predict_with_std(self.X_test)
        
        # Verify results
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(len(uncertainties), len(self.X_test))
        
        # Uncertainties should be non-negative
        self.assertTrue(np.all(uncertainties >= 0))
    
    def test_model_hyperparameter_optimization(self):
        """Test model hyperparameter optimization."""
        # Skip if data is too small
        if len(self.X_train) < 30:
            self.skipTest("Not enough data for hyperparameter optimization test")
        
        # Initialize Random Forest model
        rf_model = RandomForestModel(n_estimators=5, random_state=42)
        
        # Define simple parameter grid
        param_grid = {
            'n_estimators': [5, 10],
            'max_depth': [None, 5],
            'min_samples_split': [2, 3]
        }
        
        # Run optimization with minimal iterations
        best_params = rf_model.hyperparameter_optimize(
            self.X_train, self.y_train, param_grid, 
            method="random", n_trials=2, cv=2
        )
        
        # Verify results
        self.assertIsNotNone(best_params)
        self.assertIn('n_estimators', best_params)
        
        # Brief test of Neural Network optimization (just to check it runs)
        nn_model = NeuralNetworkModel(
            architecture={"hidden_layers": [16, 8]},
            training={"epochs": 3},
            random_state=42
        )
        
        nn_param_grid = {
            'hidden_layers': [[16, 8], [8, 4]],
            'learning_rate': [0.01, 0.001],
            'batch_size': [16]
        }
        
        # Run optimization with minimal settings
        try:
            nn_model.hyperparameter_optimize(
                self.X_train, self.y_train, nn_param_grid,
                method="random", n_trials=2, cv=2
            )
        except Exception as e:
            self.fail(f"Neural network hyperparameter optimization failed: {e}")

if __name__ == '__main__':
    unittest.main()