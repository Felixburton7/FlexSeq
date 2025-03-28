"""
Test the FlexSeq pipeline end-to-end.

This script tests the core pipeline functionality including:
- Data loading and processing
- Configuration handling and templating
- Model training and evaluation
"""

import os
import sys
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np

# Add parent directory to path to import flexseq
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexseq.config import load_config, get_enabled_models
from flexseq.pipeline import Pipeline
from flexseq.data.processor import load_and_process_data

class TestPipeline(unittest.TestCase):
    """Test the FlexSeq pipeline end-to-end."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for output and models
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'output')
        self.models_dir = os.path.join(self.temp_dir, 'models')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Base path for test data
        self.test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        
        # Check if test data exists
        self.data_path_320 = os.path.join(self.test_data_dir, 'temperature_320_train.csv')
        self.assertTrue(os.path.exists(self.data_path_320), f"Test data not found at {self.data_path_320}")
        
        # Load default config and modify for testing
        self.config = load_config()
        self.config['paths']['output_dir'] = self.output_dir
        self.config['paths']['models_dir'] = self.models_dir
        self.config['paths']['data_dir'] = self.test_data_dir
        
        # Modify config for faster testing
        self.config['models']['random_forest']['n_estimators'] = 10
        self.config['models']['neural_network']['training']['epochs'] = 5
        self.config['models']['neural_network']['training']['batch_size'] = 16
        
        # Ensure both models are enabled
        self.config['models']['random_forest']['enabled'] = True
        self.config['models']['neural_network']['enabled'] = True
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_data_loading(self):
        """Test data loading and processing."""
        # Test with temperature 320
        self.config['temperature']['current'] = 320
        
        # Load data
        df = load_and_process_data(self.data_path_320, self.config)
        
        # Assert data was loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check that required columns exist
        required_cols = ['domain_id', 'resid', 'resname', 'rmsf_320']
        for col in required_cols:
            self.assertIn(col, df.columns)
        
        # Check that window features were created if enabled
        if self.config['dataset']['features']['window']['enabled']:
            window_cols = [col for col in df.columns if '_offset_' in col]
            self.assertGreater(len(window_cols), 0)
    
    def test_temperature_templating(self):
        """Test temperature-specific configuration templating."""
        # Test with temperature 320
        config_320 = load_config(temperature=320)
        self.assertEqual(config_320['temperature']['current'], 320)
        self.assertEqual(config_320['dataset']['target'], 'rmsf_320')
        
        # Test with temperature 450
        config_450 = load_config(temperature=450)
        self.assertEqual(config_450['temperature']['current'], 450)
        self.assertEqual(config_450['dataset']['target'], 'rmsf_450')
    
    def test_pipeline_train_random_forest(self):
        """Test training a Random Forest model."""
        # Configure for random forest only
        self.config['models']['random_forest']['enabled'] = True
        self.config['models']['neural_network']['enabled'] = False
        
        # Create pipeline
        pipeline = Pipeline(self.config)
        
        # Train model
        models = pipeline.train(['random_forest'], self.data_path_320)
        
        # Assert model was trained
        self.assertIn('random_forest', models)
        self.assertIsNotNone(models['random_forest'].model)
        
        # Check model file was saved
        model_path = os.path.join(self.models_dir, 'random_forest.pkl')
        self.assertTrue(os.path.exists(model_path))
    
    def test_pipeline_predict(self):
        """Test model prediction."""
        # Configure for random forest only (faster)
        self.config['models']['random_forest']['enabled'] = True
        self.config['models']['neural_network']['enabled'] = False
        
        # Create pipeline
        pipeline = Pipeline(self.config)
        
        # Train model
        pipeline.train(['random_forest'], self.data_path_320)
        
        # Make predictions
        predictions_df = pipeline.predict(self.data_path_320, 'random_forest')
        
        # Assert predictions were made
        self.assertIsInstance(predictions_df, pd.DataFrame)
        self.assertIn('rmsf_320_predicted', predictions_df.columns)
        
        # Check prediction values are reasonable
        pred_col = 'rmsf_320_predicted'
        self.assertTrue(np.all(predictions_df[pred_col] >= 0))  # RMSF values should be positive
    
    def test_pipeline_evaluate(self):
        """Test model evaluation."""
        # Configure for random forest only (faster)
        self.config['models']['random_forest']['enabled'] = True
        self.config['models']['neural_network']['enabled'] = False
        
        # Create pipeline
        pipeline = Pipeline(self.config)
        
        # Train model
        pipeline.train(['random_forest'], self.data_path_320)
        
        # Evaluate model
        results = pipeline.evaluate(['random_forest'], self.data_path_320)
        
        # Assert evaluation was performed
        self.assertIn('random_forest', results)
        self.assertIn('rmse', results['random_forest'])
        self.assertIn('r2', results['random_forest'])
        
        # Check evaluation metrics are reasonable
        self.assertGreater(results['random_forest']['r2'], -1.0)  # R² should be reasonable
        self.assertLess(results['random_forest']['r2'], 1.1)  # R² should be <= 1.0 (with some tolerance)
    
    def test_omniflex_mode(self):
        """Test OmniFlex mode with ESM and voxel features."""
        # Configure for OmniFlex mode
        self.config['mode']['active'] = 'omniflex'
        self.config['mode']['omniflex']['use_esm'] = True
        self.config['mode']['omniflex']['use_voxel'] = True
        
        # Use random forest only (faster)
        self.config['models']['random_forest']['enabled'] = True
        self.config['models']['neural_network']['enabled'] = False
        
        # Create pipeline
        pipeline = Pipeline(self.config)
        
        # Load data
        df = load_and_process_data(self.data_path_320, self.config)
        
        # Check that ESM and voxel features are included
        self.assertIn('esm_rmsf', df.columns)
        self.assertIn('voxel_rmsf', df.columns)
        
        # Train model
        models = pipeline.train(['random_forest'], self.data_path_320)
        
        # Assert model was trained
        self.assertIn('random_forest', models)
        
        # Check feature importances include ESM and voxel
        feature_importances = models['random_forest'].get_feature_importance()
        self.assertTrue(any('esm_rmsf' in feat for feat in feature_importances.keys()))
        self.assertTrue(any('voxel_rmsf' in feat for feat in feature_importances.keys()))

if __name__ == '__main__':
    unittest.main()