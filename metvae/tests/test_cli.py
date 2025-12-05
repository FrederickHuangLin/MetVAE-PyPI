import pytest
import os
import pandas as pd
import torch
import subprocess
import numpy as np

def test_metvae_cli_outputs(test_files):
    """
    Test the MetVAE CLI command execution and verify its outputs.
    """
    # Define the command to run
    data_path, meta_path = test_files
    
    command = [
        'metvae-cli',
        '--data', str(data_path),
        '--meta', str(meta_path),
        '--continuous_covariate_keys', 'x1',
        '--categorical_covariate_keys', 'x2',
        '--latent_dim', '100',
        '--batch_size', '100',
        '--learning_rate', '0.01',
        '--sparse_method', 'pval'
    ]
    
    try:
        # Run the command and capture any output
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Check if files exist
        assert os.path.exists('df_sparse_pval.csv'), "Sparse correlation matrix file was not created"
        assert os.path.exists('model_state.pth'), "Model state file was not created"
        
    except subprocess.CalledProcessError as e:
        pytest.fail(f"CLI command failed with error: {e.stderr}")
    finally:
        # Clean up created files after testing
        for file in ['df_sparse_pval.csv', 'model_state.pth']:
            if os.path.exists(file):
                os.remove(file)

def test_metvae_cli_error_handling():
    """
    Test error handling of the MetVAE CLI command with invalid inputs.
    """
    # Test with non-existent input file
    command = [
        'metvae-cli',
        '--data', 'nonexistent.csv',
        '--meta', 'test_smd.csv'
    ]
    
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode != 0, "CLI should fail with non-existent input file"
    
    
    
    
    