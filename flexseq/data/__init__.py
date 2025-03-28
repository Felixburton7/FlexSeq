"""
Data handling modules for the FlexSeq ML pipeline.

This package contains functions for loading, processing, and
manipulating protein data across multiple temperatures.
"""

# Import key functions for easier access
from flexseq.data.loader import load_file, list_data_files, get_temperature_files
from flexseq.data.processor import (
    load_and_process_data, 
    clean_data, 
    prepare_data_for_model
)