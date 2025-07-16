# test_processor.py
import contextlib
import io
import pandas as pd
import os

from processor import TrainingProcessor, MappingProcessor

def test_find_best_ideal_functions():
    tp = TrainingProcessor(data_dir="data")
    matches = tp.find_best_ideal_functions()

    # Ensure we get 4 unique matches
    assert len(matches) == 4
    ideal_functions = [v['ideal_function'] for v in matches.values()]
    assert len(set(ideal_functions)) == 4

def test_map_test_data_creates_csv():
    tp = TrainingProcessor(data_dir="data")
    matches = tp.find_best_ideal_functions()
    mp = MappingProcessor(matches, data_dir="data")
    
    # Suppress stdout temporarily
    with contextlib.redirect_stdout(io.StringIO()):
        df = mp.map_test_data()

    # Ensure output file was created
    output_file = os.path.join("data", "mapped_test_points.csv")
    assert os.path.exists(output_file)

    # Ensure the file has the right columns
    loaded = pd.read_csv(output_file)
    assert all(col in loaded.columns for col in ["x", "y", "delta_y", "ideal_function"])

if __name__ == "__main__":
    test_find_best_ideal_functions()
    test_map_test_data_creates_csv()
    print("✅ All tests passed.")