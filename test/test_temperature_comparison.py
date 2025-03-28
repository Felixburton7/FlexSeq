"""
Test temperature comparison functionality for the FlexSeq pipeline.

This script tests:
- Loading data at multiple temperatures
- Comparing models across temperatures
- Temperature scaling analysis
- Visualization data generation
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

from flexseq.config import load_config, get_enabled_models, get_available_temperatures
from flexseq.pipeline import Pipeline
from flexseq.data.loader import get_temperature_files, load_temperature_data
from flexseq.temperature.comparison import (
    compare_temperature_predictions,
    calculate_temperature_correlations,
    generate_temperature_metrics,
    analyze_temperature_effects,
    prepare_temperature_comparison_data
)

class TestTemperatureComparison(unittest.TestCase):
    """Test temperature comparison functionality."""
    
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
        
        # Check if test data exists for multiple temperatures
        temps = [320, 450, 'average']
        self.data_paths = {}
        
        for temp in temps:
            path = os.path.join(self.test_data_dir, f'temperature_{temp}_train.csv')
            if os.path.exists(path):
                self.data_paths[temp] = path
        
        # Ensure we have at least two temperatures for comparison
        self.assertGreaterEqual(len(self.data_paths), 2, 
                               "Need at least two temperature files for comparison tests")
        
        # Load default config and modify for testing
        self.config = load_config()
        self.config['paths']['output_dir'] = self.output_dir
        self.config['paths']['models_dir'] = self.models_dir
        self.config['paths']['data_dir'] = self.test_data_dir
        
        # Modify config for faster testing
        self.config['models']['random_forest']['n_estimators'] = 10
        self.config['models']['neural_network']['enabled'] = False  # Use only RF for speed
        
        # Temperature comparison settings
        self.config['temperature']['comparison']['enabled'] = True
        
        # Create predictions for each temperature
        self.predictions = {}
        self.trained_models = {}
        
        # Train a small model and generate predictions for each temperature
        for temp, path in self.data_paths.items():
            # Update config for this temperature
            temp_config = load_config()
            temp_config['temperature']['current'] = temp
            temp_config['paths']['output_dir'] = os.path.join(self.output_dir, f"outputs_{temp}")
            temp_config['paths']['models_dir'] = os.path.join(self.models_dir, f"models_{temp}")
            temp_config['models']['random_forest']['n_estimators'] = 10
            temp_config['models']['neural_network']['enabled'] = False
            
            # Create output directories
            os.makedirs(temp_config['paths']['output_dir'], exist_ok=True)
            os.makedirs(temp_config['paths']['models_dir'], exist_ok=True)
            
            # Create pipeline
            pipeline = Pipeline(temp_config)
            
            # Train model (only for a subset of temperatures to save time)
            if temp in [320, 450]:  # Only train models for some temperatures
                models = pipeline.train(['random_forest'], path)
                self.trained_models[temp] = models
                
                # Generate predictions
                pred_df = pipeline.predict(path, 'random_forest')
                
                # Store both the DataFrame and the target column for tests
                target_col = f"rmsf_{temp}"
                self.predictions[temp] = {
                    'df': pred_df,
                    'target': target_col,
                    'predicted': f"{target_col}_predicted"
                }
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_temperature_file_detection(self):
        """Test detection of temperature-specific data files."""
        # Get temperature files
        temp_files = get_temperature_files(self.test_data_dir)
        
        # Verify we found our test files
        self.assertGreaterEqual(len(temp_files), 2)
        
        # Verify the correct temperatures were detected
        if 320 in self.data_paths:
            self.assertIn(320, temp_files)
        
        if 450 in self.data_paths:
            self.assertIn(450, temp_files)
    
    def test_load_temperature_data(self):
        """Test loading data for specific temperatures."""
        # Set available temperatures in config
        self.config['temperature']['available'] = list(self.data_paths.keys())
        
        # Test loading temperature data
        for temp in self.data_paths.keys():
            # Load data for this temperature
            df = load_temperature_data(self.config, temp)
            
            # Verify data was loaded correctly
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            
            # Check that the correct target column exists
            target_col = f"rmsf_{temp}"
            self.assertIn(target_col, df.columns)
    
    def test_compare_temperature_predictions(self):
        """Test comparing predictions across temperatures."""
        # Skip if we don't have predictions for multiple temperatures
        if len(self.predictions) < 2:
            self.skipTest("Need predictions for at least two temperatures")
        
        # Create DataFrames dictionary for the comparison function
        prediction_dfs = {temp: data['df'] for temp, data in self.predictions.items()}
        
        # Compare predictions
        combined_df = compare_temperature_predictions(prediction_dfs, self.config)
        
        # Verify results
        self.assertIsInstance(combined_df, pd.DataFrame)
        self.assertGreater(len(combined_df), 0)
        
        # Check that combined DataFrame has the expected columns
        for temp in self.predictions.keys():
            self.assertIn(f"actual_{temp}", combined_df.columns)
            self.assertIn(f"predicted_{temp}", combined_df.columns)
    
    def test_calculate_temperature_correlations(self):
        """Test calculation of temperature correlations."""
        # Skip if we don't have predictions for multiple temperatures
        if len(self.predictions) < 2:
            self.skipTest("Need predictions for at least two temperatures")
        
        # Create DataFrames dictionary for the comparison function
        prediction_dfs = {temp: data['df'] for temp, data in self.predictions.items()}
        
        # Compare predictions
        combined_df = compare_temperature_predictions(prediction_dfs, self.config)
        
        # Calculate correlations
        temperatures = list(self.predictions.keys())
        corr_df = calculate_temperature_correlations(combined_df, temperatures)
        
        # Verify results
        self.assertIsInstance(corr_df, pd.DataFrame)
        self.assertEqual(corr_df.shape, (len(temperatures), len(temperatures)))
        
        # Diagonal should be 1.0 (correlation with self)
        for temp in temperatures:
            self.assertAlmostEqual(corr_df.loc[str(temp), str(temp)], 1.0, places=5)
    
    def test_generate_temperature_metrics(self):
        """Test generation of temperature comparison metrics."""
        # Skip if we don't have predictions for multiple temperatures
        if len(self.predictions) < 2:
            self.skipTest("Need predictions for at least two temperatures")
        
        # Create a mock metrics dictionary
        metrics_by_temp = {}
        
        for temp, models in self.trained_models.items():
            metrics_by_temp[temp] = {
                'random_forest': {
                    'rmse': 0.2 + 0.1 * float(temp) / 450,  # Mock metrics that scale with temperature
                    'r2': 0.8 - 0.1 * float(temp) / 450,
                    'pearson_correlation': 0.9 - 0.05 * float(temp) / 450
                }
            }
        
        # Generate metrics comparison
        metrics_df = generate_temperature_metrics(metrics_by_temp, self.config)
        
        # Verify results
        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertGreaterEqual(len(metrics_df), len(metrics_by_temp))
        
        # Check that the DataFrame has the correct columns
        self.assertIn('temperature', metrics_df.columns)
        self.assertIn('model', metrics_df.columns)
        self.assertIn('rmse', metrics_df.columns)
        self.assertIn('r2', metrics_df.columns)
    
    def test_analyze_temperature_effects(self):
        """Test analysis of temperature effects on protein flexibility."""
        # Skip if we don't have predictions for multiple temperatures
        if len(self.predictions) < 2:
            self.skipTest("Need predictions for at least two temperatures")
        
        # Create DataFrames dictionary for the comparison function
        prediction_dfs = {temp: data['df'] for temp, data in self.predictions.items()}
        
        # Compare predictions
        combined_df = compare_temperature_predictions(prediction_dfs, self.config)
        
        # Analyze temperature effects
        temperatures = list(self.predictions.keys())
        effects = analyze_temperature_effects(combined_df, temperatures, self.config)
        
        # Verify results
        self.assertIsInstance(effects, dict)
        
        # Check that the analysis includes the expected keys
        expected_keys = ['domain_trends', 'residue_outliers', 'domain_stats', 'aa_responses']
        for key in expected_keys:
            self.assertIn(key, effects)
            self.assertIsInstance(effects[key], list)
    
    def test_prepare_temperature_comparison_data(self):
        """Test preparation of temperature comparison visualization data."""
        # Skip if we don't have predictions for multiple temperatures
        if len(self.predictions) < 2:
            self.skipTest("Need predictions for at least two temperatures")
        
        # Create comparison output directory
        comparison_dir = os.path.join(self.output_dir, "outputs_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Prepare comparison data
        try:
            comparison_data = prepare_temperature_comparison_data(
                self.config, 'random_forest', comparison_dir
            )
            
            # Verify results
            self.assertIsInstance(comparison_data, dict)
            
            # Check that output files were created
            expected_files = [
                "combined_predictions.csv",
                "actual_correlations.csv",
                "predicted_correlations.csv",
                "temperature_metrics.csv"
            ]
            
            for filename in expected_files:
                path = os.path.join(comparison_dir, filename)
                self.assertTrue(os.path.exists(path), f"File not created: {path}")
                
        except Exception as e:
            self.fail(f"prepare_temperature_comparison_data failed: {e}")

if __name__ == '__main__':
    unittest.main()