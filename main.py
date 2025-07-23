"""
main.py

Main entry point for executing the Python written exam project.

This script:
- Runs automated unit tests
- Loads and compares training data to ideal functions
- Maps test data points to best-fitting ideal functions based on deviation
- Generates an interactive Bokeh visualization of results
"""

import contextlib
import io

# Import project components
import test_processor
from processor import TrainingProcessor, MappingProcessor, Visualizer

def main():
    """
    Runs the full pipeline: testing, training-ideal matching, test mapping, and visualization.
    """
    print("Running basic unit tests...")
    
    # Step 1: Run basic unit tests and suppress internal print output
    with contextlib.redirect_stdout(io.StringIO()):
        test_processor.test_find_best_ideal_functions()
        test_processor.test_map_test_data_creates_csv()

    print("All tests passed.\n")

    # Step 2: Match training functions to best ideal functions
    tp = TrainingProcessor(data_dir="data")
    matches = tp.find_best_ideal_functions()

    # Step 3: Map test data points to ideal functions using deviation threshold
    mp = MappingProcessor(matches, data_dir="data")
    mp.map_test_data()

    # Step 4: Create visualization of all results using Bokeh
    vis = Visualizer(matches, data_dir="data", output_path="data/function_mapping_visualization.html")
    vis.create_plot()

if __name__ == "__main__":
    main()
