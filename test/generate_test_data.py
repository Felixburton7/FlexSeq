"""
Generate synthetic protein data for testing the FlexSeq pipeline.

This script creates test data files at different temperatures with realistic
protein flexibility patterns for testing the FlexSeq ML pipeline.
"""

import os
import numpy as np
import pandas as pd

# Create test_data directory if it doesn't exist
os.makedirs('test/test_data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for test data
n_domains = 5  # Number of protein domains
residues_per_domain = 50  # Residues per domain
temperatures = [320, 450, "average"]  # Temperatures to generate data for

# Define domains
domains = [f"test{i}A00" for i in range(1, n_domains+1)]

# Amino acids and their properties
amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

# Secondary structure types
ss_types = ['H', 'E', 'C', 'T', 'G']  # Helix, Sheet, Coil, Turn, 3-10 Helix

# Surface exposure
locations = ['core', 'exterior']

# Temperature scaling factors (relative to 320K)
temp_scaling = {
    320: 1.0,
    450: 1.8,  # Higher temperature has higher flexibility
    "average": 1.4  # Average between temperatures
}

# Generate data
all_data = []

for domain in domains:
    # Domain size
    protein_size = residues_per_domain + np.random.randint(-5, 6)  # Variation in domain size
    
    # Generate secondary structure segments
    # We'll create realistic segments rather than random assignment
    ss_segments = []
    pos = 0
    while pos < protein_size:
        # Randomly choose segment type with biases
        if np.random.random() < 0.4:  # 40% chance of helix
            ss_type = 'H'
            length = np.random.randint(5, 15)  # Helices are 5-15 residues
        elif np.random.random() < 0.3:  # 30% chance of sheet
            ss_type = 'E'
            length = np.random.randint(3, 8)  # Sheets are 3-8 residues
        else:  # 30% chance of loop/turn
            if np.random.random() < 0.5:
                ss_type = 'C'  # Coil
            else:
                ss_type = 'T'  # Turn
            length = np.random.randint(2, 7)  # Loops are 2-7 residues
        
        # Add segment
        ss_segments.append((ss_type, min(length, protein_size - pos)))
        pos += length
    
    # Flatten secondary structure assignment
    secondary_structure = []
    for ss_type, length in ss_segments:
        secondary_structure.extend([ss_type] * length)
    
    # Truncate if needed
    secondary_structure = secondary_structure[:protein_size]
    
    # Generate data for each residue
    for resid in range(protein_size):
        normalized_resid = resid / protein_size
        
        # Secondary structure for this residue
        dssp = secondary_structure[resid]
        
        # Amino acid (with some bias based on secondary structure)
        if dssp == 'H':  # Helices favor certain amino acids
            weights = [2 if aa in ['ALA', 'LEU', 'GLU', 'LYS'] else 1 for aa in amino_acids]
        elif dssp == 'E':  # Sheets favor certain amino acids
            weights = [2 if aa in ['VAL', 'ILE', 'THR', 'TYR'] else 1 for aa in amino_acids]
        else:  # Loops/turns favor certain amino acids
            weights = [2 if aa in ['GLY', 'PRO', 'ASN', 'ASP'] else 1 for aa in amino_acids]
            
        weights = np.array(weights)
        resname = np.random.choice(amino_acids, p=weights/weights.sum())
        
        # Surface exposure (more likely to be exterior for loops)
        if dssp in ['C', 'T']:
            core_exterior = np.random.choice(locations, p=[0.3, 0.7])
        else:
            core_exterior = np.random.choice(locations, p=[0.6, 0.4])
            
        # Generate relative accessibility based on core/exterior
        if core_exterior == 'core':
            relative_accessibility = np.random.uniform(0, 0.3)
        else:
            relative_accessibility = np.random.uniform(0.3, 1.0)
        
        # Generate dihedral angles based on secondary structure
        if dssp == 'H':  # Alpha helix
            phi = np.random.normal(-57, 10)
            psi = np.random.normal(-47, 10)
        elif dssp == 'E':  # Beta sheet
            phi = np.random.normal(-140, 15)
            psi = np.random.normal(135, 15)
        elif dssp == 'T':  # Turn
            phi = np.random.normal(-60, 30)
            psi = np.random.normal(30, 30)
        elif dssp == 'G':  # 3-10 helix
            phi = np.random.normal(-74, 10)
            psi = np.random.normal(-4, 10)
        else:  # Coil
            phi = np.random.normal(-60, 40)
            psi = np.random.normal(0, 40)
        
        # Normalize dihedral angles
        phi_norm = (phi % 360) / 180 - 1
        psi_norm = (psi % 360) / 180 - 1
        
        # Encode categorical variables
        resname_encoded = amino_acids.index(resname) + 1
        core_exterior_encoded = 0 if core_exterior == 'core' else 1
        secondary_structure_encoded = 0 if dssp in ['H', 'G'] else (1 if dssp == 'E' else 2)
        
        # Generate base flexibility 
        # RMSF is higher for:
        # - Loops (dssp = C, T)
        # - Surface residues (core_exterior = exterior)
        # - Terminal residues (low or high normalized_resid)
        # - Certain amino acids (GLY, PRO)
        
        base_rmsf = 0.5  # Base RMSF value
        
        # Secondary structure contribution
        if dssp in ['H', 'G']:  # Helices are more rigid
            ss_factor = 0.7
        elif dssp == 'E':  # Sheets are also rigid
            ss_factor = 0.8
        else:  # Loops/turns are flexible
            ss_factor = 1.5
            
        # Surface exposure contribution
        exposure_factor = 1.0 if core_exterior == 'core' else 1.4
        
        # Terminal position contribution (U-shaped curve)
        position_factor = 1.0 + 1.0 * ((2 * normalized_resid - 1) ** 2)
        
        # Amino acid contribution
        if resname == 'GLY':  # Glycine is very flexible
            aa_factor = 1.5
        elif resname == 'PRO':  # Proline is rigid but can create kinks
            aa_factor = 1.2
        else:
            aa_factor = 1.0
            
        # Optional ESM and voxel predictions for OmniFlex mode
        esm_rmsf = base_rmsf * ss_factor * exposure_factor * position_factor * aa_factor * np.random.normal(1.0, 0.2)
        voxel_rmsf = base_rmsf * ss_factor * exposure_factor * position_factor * aa_factor * np.random.normal(1.0, 0.2)
        
        # Base RMSF for multiple temperatures
        for temp in temperatures:
            # Calculate RMSF with temperature scaling and some random noise
            scaling = temp_scaling[temp]
            rmsf = base_rmsf * ss_factor * exposure_factor * position_factor * aa_factor * scaling
            
            # Add some random variation
            rmsf *= np.random.normal(1.0, 0.1)
            
            # Format the temperature for the column name
            temp_str = str(temp)
            rmsf_col = f"rmsf_{temp_str}"
            
            # Create a data row
            row = {
                'domain_id': domain,
                'resid': resid,
                'resname': resname,
                rmsf_col: rmsf,
                'protein_size': protein_size,
                'normalized_resid': normalized_resid,
                'core_exterior': core_exterior,
                'relative_accessibility': relative_accessibility,
                'dssp': dssp,
                'phi': phi,
                'psi': psi,
                'resname_encoded': resname_encoded,
                'core_exterior_encoded': core_exterior_encoded,
                'secondary_structure_encoded': secondary_structure_encoded,
                'phi_norm': phi_norm,
                'psi_norm': psi_norm,
                'esm_rmsf': esm_rmsf,
                'voxel_rmsf': voxel_rmsf
            }
            
            all_data.append(row)

# Create the combined dataframe
df = pd.DataFrame(all_data)

# Split data by temperature and save
for temp in temperatures:
    temp_str = str(temp)
    temp_cols = [col for col in df.columns if col != 'rmsf_320' and col != 'rmsf_450' and col != 'rmsf_average' 
                 or col == f'rmsf_{temp_str}']
    
    # Select only columns for this temperature
    temp_df = df[temp_cols].copy()
    
    # Save to CSV
    output_path = f'test/test_data/temperature_{temp_str}_train.csv'
    temp_df.to_csv(output_path, index=False)
    print(f"Created test data file: {output_path} with {len(temp_df)} rows")

print("Test data generation complete!")