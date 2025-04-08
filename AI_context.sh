#!/bin/bash

# Define the output file
OUTPUT_FILE="$PWD/FlexSeq_context.txt"

{
    echo "=========================================================="
    echo "                FlexSeq: Protein Flexibility ML Pipeline"
    echo "=========================================================="
    echo ""
    
    # Example data format (now at the top)
    echo "=========================================================="
    echo "Example Input Data Format"
    echo "=========================================================="
    echo "FlexSeq expects temperature-specific CSV files in the data directory."
    echo "Example file name format: 'temperature_320_train.csv'"
    echo ""
    echo "Expected columns in the CSV:"
    echo "- domain_id: Protein domain identifier (e.g., '1a0aA00')"
    echo "- resid: Residue ID (integer position in the protein chain)"
    echo "- resname: Amino acid type (e.g., ALA, LYS, etc.)"
    echo "- rmsf_{temperature}: Target RMSF value at the specified temperature"
    echo "- protein_size: Total number of residues in the protein"
    echo "- normalized_resid: Position normalized to 0-1 range"
    echo "- core_exterior: Location classification ('interior' or 'surface')"
    echo "- relative_accessibility: Solvent accessibility measure (0-1)"
    echo "- dssp: Secondary structure annotation (H=helix, E=sheet, C=coil, etc.)"
    echo "- phi, psi: Backbone dihedral angles"
    echo "- *_encoded: Various numerically encoded categorical features"
    echo "- phi_norm, psi_norm: Normalized dihedral angles to [-1, 1] range"
    echo ""
    echo "For OmniFlex mode, additional columns:"
    echo "- esm_rmsf: Predictions from ESM embeddings"
    echo "- voxel_rmsf: Predictions from 3D voxel representation"
    echo ""
    
    # Usage examples (now at the top)
    echo "=========================================================="
    echo "Usage Examples"
    echo "=========================================================="
    echo "# Training a model at a specific temperature"
    echo "flexseq train --temperature 320"
    echo ""
    echo "# Training models on all available temperatures"
    echo "flexseq train-all-temps"
    echo ""
    echo "# Evaluate a trained model"
    echo "flexseq evaluate --model random_forest --temperature 320"
    echo ""
    echo "# Generate predictions using the best model"
    echo "flexseq predict --input new_proteins.csv --temperature 320"
    echo ""
    echo "# Compare results across temperatures"
    echo "flexseq compare-temperatures"
    echo ""
    echo "# Use OmniFlex mode with advanced features"
    echo "flexseq train --mode omniflex --temperature 320"
    echo ""
    
    echo "Project Working Directory: $(pwd)"
    echo ""
    
    # Print project structure
    echo "Project Tree Structure:"
    echo "---------------------------------------------------------"
    find . -type d -name "__pycache__" -prune -o -type d -path "*/\.*" -prune -o -path "./data" -prune -o -path "./output" -prune -o -path "./models" -prune -o -path "./test" -prune -o -type d -print | sort
    echo ""
    
    # Print file listing (excluding data, cache, and output directories)
    echo "File Listing (excluding cache, data, output, and test directories):"
    echo "---------------------------------------------------------"
    find . -type d -name "__pycache__" -prune -o -type d -path "*/\.*" -prune -o -path "./data" -prune -o -path "./output" -prune -o -path "./models" -prune -o -path "./test" -prune -o -type f -name "*.py" -o -name "*.yaml" -o -name "*.toml" | sort
    echo ""
    
    # Print default configuration
    echo "=========================================================="
    echo "Default Configuration (default_config.yaml)"
    echo "=========================================================="
    cat default_config.yaml
    echo ""
    
    # Print package structure and files
    echo "=========================================================="
    echo "FlexSeq Package Files"
    echo "=========================================================="
    
    # Main package files
    echo "### Main Package Files ###"
    echo "---------------------------------------------------------"
    for file in pyproject.toml setup.py README.md; do
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done
    
    # Core module files
    echo "### Core Module Files ###"
    echo "---------------------------------------------------------"
    for file in flexseq/__init__.py flexseq/config.py flexseq/pipeline.py flexseq/cli.py; do
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done
    
    # Model files
    echo "### Model Files ###"
    echo "---------------------------------------------------------"
    for file in flexseq/models/__init__.py flexseq/models/base.py flexseq/models/random_forest.py flexseq/models/neural_network.py; do
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done
    
    # Data handling files
    echo "### Data Handling Files ###"
    echo "---------------------------------------------------------"
    for file in flexseq/data/__init__.py flexseq/data/loader.py flexseq/data/processor.py; do
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done
    
    # Temperature handling files
    echo "### Temperature Handling Files ###"
    echo "---------------------------------------------------------"
    for file in flexseq/temperature/__init__.py flexseq/temperature/comparison.py; do
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done
    
    # Utility files
    echo "### Utility Files ###"
    echo "---------------------------------------------------------"
    for file in flexseq/utils/__init__.py flexseq/utils/helpers.py flexseq/utils/metrics.py flexseq/utils/visualization.py; do
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done
    
    # Output utils files
    echo "### Output Utils Files ###"
    echo "---------------------------------------------------------"
    for file in output/utils/__init__.py output/utils/helpers.py output/utils/metrics.py output/utils/visualization.py; do
        if [ -f "$file" ]; then
            echo "===== FILE: $file ====="
            cat "$file"
            echo ""
        fi
    done
    
    # NEW SECTION: Output result files with first 15 lines
    echo "=========================================================="
    echo "Output Result Files (First 15 lines of each file)"
    echo "=========================================================="
    echo ""
    
    # Find all files in the output directory, excluding pngs and the utils directory
    echo "Finding output result files..."
    output_files=$(find ./output -type f -not -name "*.png" -not -path "*/utils/*" -not -path "*/\.*" 2>/dev/null | sort)
    
    if [ -z "$output_files" ]; then
        echo "No output result files found."
    else
        for file in $output_files; do
            echo "===== FILE: $file ====="
            echo "First 15 lines:"
            head -15 "$file"
            echo ""
            echo "---------------------------------------------------------"
        done
    fi
    
    echo "=========================================================="
    echo "End of FlexSeq Context Document"
    echo "=========================================================="

} > "$OUTPUT_FILE"

echo "Context file created at: $OUTPUT_FILE"