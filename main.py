import contextlib
import io

import test_processor
from processor import TrainingProcessor, MappingProcessor, Visualizer

def main():
    # Run integrated tests first
    print("🧪 Running basic unit tests...")
    
    # Suppress any print output during tests
    with contextlib.redirect_stdout(io.StringIO()):
        test_processor.test_find_best_ideal_functions()
        test_processor.test_map_test_data_creates_csv()

    print("✅ All tests passed.\n")

    # Continue with processing
    tp = TrainingProcessor(data_dir="data")
    matches = tp.find_best_ideal_functions()

    mp = MappingProcessor(matches, data_dir="data")
    mp.map_test_data()

    vis = Visualizer(matches, data_dir="data", output_path="data/function_mapping_visualization.html")
    vis.create_plot()

if __name__ == "__main__":
    main()