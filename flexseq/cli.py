"""
Command-line interface for the FlexSeq ML pipeline.

This module provides the CLI commands for training, evaluating, and
analyzing protein flexibility predictions across multiple temperatures.
"""

import os
import sys
import logging
from typing import List, Optional, Tuple, Dict, Any, Union

import click

from flexseq.config import (
    load_config, 
    get_enabled_models, 
    get_model_config,
    get_available_temperatures,
    get_output_dir_for_temperature,
    get_models_dir_for_temperature,
    get_comparison_output_dir
)
from flexseq.pipeline import Pipeline
from flexseq.models import get_available_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_model_list(model_arg: Optional[str]) -> List[str]:
    """
    Parse comma-separated list of models.
    
    Args:
        model_arg: Comma-separated model names or None
        
    Returns:
        List of model names
    """
    if not model_arg:
        return []
        
    return [m.strip() for m in model_arg.split(",")]

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    FlexSeq: ML pipeline for protein flexibility prediction.
    
    This tool provides a complete pipeline for predicting protein flexibility
    (RMSF values) from sequence and structural features using machine learning
    across multiple temperatures.
    """
    pass

@cli.command()
@click.option("--model", 
              help="Model to train (comma-separated for multiple)")
@click.option("--config", 
              type=click.Path(exists=True), 
              help="Path to config file")
@click.option("--param", 
              multiple=True, 
              help="Override config parameter (e.g. models.random_forest.n_estimators=200)")
@click.option("--domains", 
              help="Specific domains to include (comma-separated)")
@click.option("--exclude-domains", 
              help="Domains to exclude (comma-separated)")
@click.option("--disable-feature", 
              help="Features to disable (comma-separated)")
@click.option("--window-size", 
              type=int, 
              help="Window size for feature engineering")
@click.option("--input", 
              type=click.Path(exists=True), 
              help="Input data file (CSV)")
@click.option("--temperature", 
              type=str,
              help="Temperature to use (e.g., 320, 348, average)")
@click.option("--mode",
              type=click.Choice(["flexseq", "omniflex"]),
              help="Operation mode")
def train(
    model, config, param, domains, exclude_domains, 
    disable_feature, window_size, input, temperature, mode
):
    """
    Train flexibility prediction models.
    
    Examples:
        flexseq train
        flexseq train --model random_forest
        flexseq train --temperature 320
        flexseq train --mode omniflex
    """
    # Load configuration
    cfg = load_config(config, param, temperature=temperature)
    
    # Set mode if specified
    if mode:
        cfg["mode"]["active"] = mode
    
    # Apply CLI-specific overrides
    if domains:
        domain_list = [d.strip() for d in domains.split(",")]
        cfg["dataset"]["domains"]["include"] = domain_list
        
    if exclude_domains:
        exclude_list = [d.strip() for d in exclude_domains.split(",")]
        cfg["dataset"]["domains"]["exclude"] = exclude_list
        
    if disable_feature:
        features = [f.strip() for f in disable_feature.split(",")]
        for feature in features:
            if feature in cfg["dataset"]["features"]["use_features"]:
                cfg["dataset"]["features"]["use_features"][feature] = False
                
    if window_size is not None:
        cfg["dataset"]["features"]["window"]["size"] = window_size
    
    # Determine which models to train
    model_list = parse_model_list(model)
    if not model_list:
        model_list = get_enabled_models(cfg)
        
    if not model_list:
        click.echo("No models specified or enabled in config")
        return
    
    # Create temperature-specific output directory
    current_temp = cfg["temperature"]["current"]
    output_dir = get_output_dir_for_temperature(cfg, current_temp)
    models_dir = get_models_dir_for_temperature(cfg, current_temp)
    
    # Update config with temperature-specific directories
    cfg["paths"]["output_dir"] = output_dir
    cfg["paths"]["models_dir"] = models_dir
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Create pipeline and train models
    pipeline = Pipeline(cfg)
    
    try:
        trained_models = pipeline.train(model_list, input)
        click.echo(f"Successfully trained {len(trained_models)} models for temperature {current_temp}")
    except Exception as e:
        click.echo(f"Error during training: {e}")
        sys.exit(1)

@cli.command()
@click.option("--model", 
              help="Model to evaluate (comma-separated for multiple)")
@click.option("--config", 
              type=click.Path(exists=True), 
              help="Path to config file")
@click.option("--param", 
              multiple=True, 
              help="Override config parameter")
@click.option("--input", 
              type=click.Path(exists=True), 
              help="Input data file (CSV)")
@click.option("--temperature", 
              type=str,
              help="Temperature to use (e.g., 320, 348, average)")
@click.option("--mode",
              type=click.Choice(["flexseq", "omniflex"]),
              help="Operation mode")
def evaluate(model, config, param, input, temperature, mode):
    """
    Evaluate trained models.
    
    Examples:
        flexseq evaluate
        flexseq evaluate --model random_forest
        flexseq evaluate --temperature 320
        flexseq evaluate --mode omniflex
    """
    # Load configuration
    cfg = load_config(config, param, temperature=temperature)
    
    # Set mode if specified
    if mode:
        cfg["mode"]["active"] = mode
    
    # Determine which models to evaluate
    model_list = parse_model_list(model)
    if not model_list:
        model_list = get_enabled_models(cfg)
        
    if not model_list:
        click.echo("No models specified or enabled in config")
        return
    
    # Get temperature-specific directories
    current_temp = cfg["temperature"]["current"]
    output_dir = get_output_dir_for_temperature(cfg, current_temp)
    models_dir = get_models_dir_for_temperature(cfg, current_temp)
    
    # Update config with temperature-specific directories
    cfg["paths"]["output_dir"] = output_dir
    cfg["paths"]["models_dir"] = models_dir
    
    # Create pipeline and evaluate models
    pipeline = Pipeline(cfg)
    
    try:
        results = pipeline.evaluate(model_list, input)
        
        # Display results
        click.echo("\nEvaluation Results:")
        for model_name, metrics in results.items():
            click.echo(f"\n{model_name}:")
            for metric, value in metrics.items():
                click.echo(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        click.echo(f"Error during evaluation: {e}")
        sys.exit(1)

@cli.command()
@click.option("--model", 
              help="Model to use (defaults to best model)")
@click.option("--config", 
              type=click.Path(exists=True), 
              help="Path to config file")
@click.option("--param", 
              multiple=True, 
              help="Override config parameter")
@click.option("--input", 
              type=click.Path(exists=True), 
              required=True,
              help="Input data file (CSV)")
@click.option("--output", 
              type=click.Path(), 
              help="Output file path (defaults to input_predictions.csv)")
@click.option("--temperature", 
              type=str,
              help="Temperature to use (e.g., 320, 348, average)")
@click.option("--mode",
              type=click.Choice(["flexseq", "omniflex"]),
              help="Operation mode")
@click.option("--uncertainty",
              is_flag=True,
              help="Include uncertainty estimates in predictions")
def predict(model, config, param, input, output, temperature, mode, uncertainty):
    """
    Generate predictions for new data.
    
    Examples:
        flexseq predict --input new_proteins.csv
        flexseq predict --model random_forest --input new_proteins.csv
        flexseq predict --temperature 320 --input new_proteins.csv
        flexseq predict --mode omniflex --input new_proteins.csv
        flexseq predict --uncertainty --input new_proteins.csv
    """
    # Load configuration
    cfg = load_config(config, param, temperature=temperature)
    
    # Set mode if specified
    if mode:
        cfg["mode"]["active"] = mode
    
    # Get temperature-specific directories
    current_temp = cfg["temperature"]["current"]
    output_dir = get_output_dir_for_temperature(cfg, current_temp)
    models_dir = get_models_dir_for_temperature(cfg, current_temp)
    
    # Update config with temperature-specific directories
    cfg["paths"]["output_dir"] = output_dir
    cfg["paths"]["models_dir"] = models_dir
    
    # Create pipeline
    pipeline = Pipeline(cfg)
    
    try:
        # Generate predictions
        predictions_df = pipeline.predict(input, model, with_uncertainty=uncertainty)
        
        # Determine output path
        if not output:
            base = os.path.splitext(input)[0]
            output = f"{base}_predictions_{current_temp}.csv"
            
        # Save predictions
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        predictions_df.to_csv(output, index=False)
        click.echo(f"Saved predictions to {output}")
        
    except Exception as e:
        click.echo(f"Error generating predictions: {e}")
        sys.exit(1)

@cli.command()
@click.option("--model", 
              help="Model to train (comma-separated for multiple)")
@click.option("--config", 
              type=click.Path(exists=True), 
              help="Path to config file")
@click.option("--param", 
              multiple=True, 
              help="Override config parameter")
@click.option("--mode",
              type=click.Choice(["flexseq", "omniflex"]),
              help="Operation mode")
def train_all_temps(model, config, param, mode):
    """
    Train models on all available temperatures.
    
    Examples:
        flexseq train-all-temps
        flexseq train-all-temps --model random_forest
        flexseq train-all-temps --mode omniflex
    """
    # Load configuration without temperature override
    cfg = load_config(config, param)
    
    # Set mode if specified
    if mode:
        cfg["mode"]["active"] = mode
    
    # Get all available temperatures
    temperatures = get_available_temperatures(cfg)
    
    # Determine which models to train
    model_list = parse_model_list(model)
    if not model_list:
        model_list = get_enabled_models(cfg)
        
    if not model_list:
        click.echo("No models specified or enabled in config")
        return
    
    # Train for each temperature
    for temp in temperatures:
        click.echo(f"\nTraining for temperature: {temp}")
        
        # Create temperature-specific config
        temp_cfg = load_config(config, param, temperature=temp)
        
        if mode:
            temp_cfg["mode"]["active"] = mode
        
        # Get temperature-specific directories
        output_dir = get_output_dir_for_temperature(temp_cfg, temp)
        models_dir = get_models_dir_for_temperature(temp_cfg, temp)
        
        # Update config with temperature-specific directories
        temp_cfg["paths"]["output_dir"] = output_dir
        temp_cfg["paths"]["models_dir"] = models_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Create pipeline and train models
        pipeline = Pipeline(temp_cfg)
        
        try:
            trained_models = pipeline.train(model_list)
            click.echo(f"Successfully trained {len(trained_models)} models for temperature {temp}")
            
            # Evaluate models
            results = pipeline.evaluate(model_list)
            
            # Display results
            click.echo(f"Evaluation Results for temperature {temp}:")
            for model_name, metrics in results.items():
                click.echo(f"{model_name}:")
                for metric, value in metrics.items():
                    click.echo(f"  {metric}: {value:.4f}")
                    
        except Exception as e:
            click.echo(f"Error during training for temperature {temp}: {e}")
            # Continue with next temperature

@cli.command()
@click.option("--model", 
              help="Model to compare (defaults to random_forest)")
@click.option("--config", 
              type=click.Path(exists=True), 
              help="Path to config file")
@click.option("--param", 
              multiple=True, 
              help="Override config parameter")
@click.option("--mode",
              type=click.Choice(["flexseq", "omniflex"]),
              help="Operation mode")
def compare_temperatures(model, config, param, mode):
    """
    Compare results across temperatures.
    
    Examples:
        flexseq compare-temperatures
        flexseq compare-temperatures --model neural_network
    """
    from flexseq.temperature.comparison import prepare_temperature_comparison_data
    
    # Load configuration
    cfg = load_config(config, param)
    
    # Set mode if specified
    if mode:
        cfg["mode"]["active"] = mode
    
    # Get model to use
    if not model:
        model = "random_forest"  # Default to random forest
        click.echo(f"No model specified, using {model}")
    
    # Get all available temperatures
    temperatures = get_available_temperatures(cfg)
    
    # Check if temperature comparison is enabled
    if not cfg["temperature"]["comparison"]["enabled"]:
        click.echo("Temperature comparison is disabled in config")
        return
    
    # Get comparison output directory
    comparison_dir = get_comparison_output_dir(cfg)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Generate comparison data
    try:
        comparison_data = prepare_temperature_comparison_data(cfg, model, comparison_dir)
        
        click.echo(f"\nTemperature comparison data saved to {comparison_dir}")
        
        # Display available files
        click.echo("Generated files:")
        for filename in os.listdir(comparison_dir):
            file_path = os.path.join(comparison_dir, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                click.echo(f"  {filename} ({file_size} bytes)")
        
    except Exception as e:
        click.echo(f"Error during temperature comparison: {e}")
        sys.exit(1)

@cli.command()
@click.option("--input", 
              type=click.Path(exists=True), 
              required=True,
              help="Input data file (CSV)")
@click.option("--config", 
              type=click.Path(exists=True), 
              help="Path to config file")
@click.option("--param", 
              multiple=True, 
              help="Override config parameter")
@click.option("--output", 
              type=click.Path(), 
              help="Output file path (defaults to input_processed.csv)")
@click.option("--temperature", 
              type=str,
              help="Temperature to use (e.g., 320, 348, average)")
@click.option("--mode",
              type=click.Choice(["flexseq", "omniflex"]),
              help="Operation mode")
def preprocess(input, config, param, output, temperature, mode):
    """
    Preprocess data only without training or prediction.
    
    Examples:
        flexseq preprocess --input raw_data.csv
        flexseq preprocess --temperature 320 --input raw_data.csv
        flexseq preprocess --mode omniflex --input raw_data.csv
    """
    from flexseq.data.processor import load_and_process_data
    
    # Load configuration
    cfg = load_config(config, param, temperature=temperature)
    
    # Set mode if specified
    if mode:
        cfg["mode"]["active"] = mode
    
    try:
        # Process data
        processed_df = load_and_process_data(input, cfg)
        
        # Determine output path
        if not output:
            base = os.path.splitext(input)[0]
            current_temp = cfg["temperature"]["current"]
            output = f"{base}_processed_{current_temp}.csv"
            
        # Save processed data
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        processed_df.to_csv(output, index=False)
        click.echo(f"Saved processed data to {output}")
        
    except Exception as e:
        click.echo(f"Error preprocessing data: {e}")
        sys.exit(1)

@cli.command()
@click.option("--model", 
              help="Model to train (comma-separated for multiple)")
@click.option("--config", 
              type=click.Path(exists=True), 
              help="Path to config file")
@click.option("--param", 
              multiple=True, 
              help="Override config parameter")
@click.option("--input", 
              type=click.Path(exists=True), 
              help="Input data file (CSV)")
@click.option("--temperature", 
              type=str,
              help="Temperature to use (e.g., 320, 348, average)")
@click.option("--mode",
              type=click.Choice(["flexseq", "omniflex"]),
              help="Operation mode")
@click.option("--skip-visualization", 
              is_flag=True,
              help="Skip visualization steps")
def run(model, config, param, input, temperature, mode, skip_visualization):
    """
    Run the complete pipeline (train, evaluate, analyze).
    
    Examples:
        flexseq run
        flexseq run --model random_forest
        flexseq run --temperature 320
        flexseq run --mode omniflex
    """
    # Load configuration
    cfg = load_config(config, param, temperature=temperature)
    
    # Set mode if specified
    if mode:
        cfg["mode"]["active"] = mode
    
    # Determine which models to use
    model_list = parse_model_list(model)
    if not model_list:
        model_list = get_enabled_models(cfg)
        
    if not model_list:
        click.echo("No models specified or enabled in config")
        return
    
    # Get temperature-specific directories
    current_temp = cfg["temperature"]["current"]
    output_dir = get_output_dir_for_temperature(cfg, current_temp)
    models_dir = get_models_dir_for_temperature(cfg, current_temp)
    
    # Update config with temperature-specific directories
    cfg["paths"]["output_dir"] = output_dir
    cfg["paths"]["models_dir"] = models_dir
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Create pipeline and run
    pipeline = Pipeline(cfg)
    
    try:
        results = pipeline.run_pipeline(
            model_list, input, skip_visualization
        )
        
        click.echo("\nPipeline completed successfully!")
        click.echo(f"Results saved to {output_dir}")
        
    except Exception as e:
        click.echo(f"Error running pipeline: {e}")
        sys.exit(1)

@cli.command()
def list_models():
    """
    List available models in the registry.
    
    Examples:
        flexseq list-models
    """
    from flexseq.models import get_available_models
    
    models = get_available_models()
    
    click.echo("Available models:")
    for model in models:
        click.echo(f"  - {model}")

@cli.command()
@click.option("--config", 
              type=click.Path(exists=True), 
              help="Path to config file")
def list_temperatures(config):
    """
    List available temperatures in the configuration.
    
    Examples:
        flexseq list-temperatures
    """
    # Load configuration
    cfg = load_config(config)
    
    temperatures = get_available_temperatures(cfg)
    
    click.echo("Available temperatures:")
    for temp in temperatures:
        click.echo(f"  - {temp}")

if __name__ == "__main__":
    cli()