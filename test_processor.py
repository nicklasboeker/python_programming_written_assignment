"""
test_processor.py

Basic unit tests for the TrainingProcessor and MappingProcessor classes.

Tests:
- Whether the training processor correctly matches each training function to a unique ideal function
- Whether the mapping processor generates a CSV file with valid structure
"""

import contextlib
import io
import pandas as pd
import os

from processor import TrainingProcessor, MappingProcessor

def test_find_best_ideal_functions():
    """
    Test that the training processor:
    - Returns exactly 4 matches
    - Maps each training function to a unique ideal function
    """
    tp = TrainingProcessor(data_dir="data")
    matches = tp.find_best_ideal_functions()

    assert len(matches) == 4
    ideal_functions = [v['ideal_function'] for v in matches.values()]
    assert len(set(ideal_functions)) == 4

def test_map_test_data_creates_csv():
    """
    Test that the mapping processor:
    - Successfully maps test points
    - Creates the correct output CSV file
    - Includes the required columns in the file
    """
    tp = TrainingProcessor(data_dir="data")
    matches = tp.find_best_ideal_functions()
    mp = MappingProcessor(matches, data_dir="data")
    
    # Suppress [INFO] print
    with contextlib.redirect_stdout(io.StringIO()):
        df = mp.map_test_data()

    output_file = os.path.join("data", "mapped_test_points.csv")
    assert os.path.exists(output_file)

    loaded = pd.read_csv(output_file)
    assert all(col in loaded.columns for col in ["x", "y", "delta_y", "ideal_function"])

if __name__ == "__main__":
    test_find_best_ideal_functions()
    test_map_test_data_creates_csv()
    print("All tests passed.")