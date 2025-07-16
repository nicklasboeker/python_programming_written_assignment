"""
processor.py

This module defines classes for processing and analyzing training, ideal, and test datasets
used in the Python written exam assignment. It includes functionality for:

- Matching training functions to ideal functions using least squares
- Mapping test data points based on deviation thresholds
- Visualizing results using Bokeh

Classes:
- FunctionProcessor: Base class for loading datasets
- TrainingProcessor: Selects best ideal functions for training data
- MappingProcessor: Maps test data to ideal functions
- Visualizer: Renders visual output as interactive HTML plot
"""

import pandas as pd
import numpy as np
import os
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Category10

class DataLoadError(Exception):
    """Custom exception for errors during data loading."""
    pass

# FUNCTION PROCESSOR
class FunctionProcessor:
    """
    Base class responsible for loading and managing datasets.

    Attributes:
        data_dir (str): Path to the folder containing the CSV files.
        train_df (DataFrame): Loaded training data.
        ideal_df (DataFrame): Loaded ideal functions.
        test_df (DataFrame): Loaded test dataset.
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.train_df = None
        self.ideal_df = None
        self.test_df = None

    def load_csv(self, filename):
        """
        Loads a single CSV file into a DataFrame.

        Args:
            filename (str): The name of the CSV file to load.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            DataLoadError: If the file does not exist.
        """
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise DataLoadError(f"File not found: {path}")
        return pd.read_csv(path)

    def load_all(self):
        """
        Loads all required CSV files (train, ideal, test) into instance variables.

        Raises:
            DataLoadError: If any file is missing or invalid.
        """
        try:
            self.train_df = self.load_csv("train.csv")
            self.ideal_df = self.load_csv("ideal.csv")
            self.test_df = self.load_csv("test.csv")
        except DataLoadError as e:
            print(f"[ERROR] {e}")
            raise

# TRAINING PROCESSOR
class TrainingProcessor(FunctionProcessor):
    """
    Inherits from FunctionProcessor.
    Responsible for comparing training functions with ideal functions to find best matches
    using the least squares method.

    Attributes:
        chosen_matches (dict): Stores the mapping from training function to ideal function and MSE.
    """

    def __init__(self, data_dir="data"):
        super().__init__(data_dir)
        self.chosen_matches = {}

    def find_best_ideal_functions(self):
        """
        Compares each of the 4 training functions (y1–y4) to all 50 ideal functions
        and selects the one with the lowest mean squared error.

        Returns:
            dict: Format like {'y1': {'ideal_function': 'y42', 'mse': 0.085}, ...}
        """
        self.load_all()
        used = set()

        for train_col in ['y1', 'y2', 'y3', 'y4']:
            min_error = float('inf')
            best_match = None

            for ideal_col in [f'y{i}' for i in range(1, 51)]:
                if ideal_col in used:
                    continue # Skip ideal functions that were already used for another training function
                error = np.mean((self.train_df[train_col] - self.ideal_df[ideal_col]) ** 2) # Calculate mean squared error between training and ideal function
                if error < min_error:
                    min_error = error
                    best_match = ideal_col

            self.chosen_matches[train_col] = {
                "ideal_function": best_match,
                "mse": min_error
            }
            used.add(best_match)

        return self.chosen_matches
    
# MAPPING PROCESSOR
class MappingProcessor(FunctionProcessor):
    """
    Handles mapping test data points to the previously selected ideal functions,
    based on a maximum deviation threshold derived from training error.

    Attributes:
        matches (dict): Chosen training→ideal function mappings with MSE
        max_devs (dict): Max deviation * sqrt(2) thresholds for mapping
    """

    def __init__(self, matches, data_dir="data"):
        super().__init__(data_dir)
        self.matches = matches
        self.max_devs = {}

    def compute_thresholds(self):
        """
        Computes the maximum deviation for each training-ideal pair and multiplies it
        by √2 to set the threshold for accepting test points.
        """
        self.load_all()
        for train_col, info in self.matches.items():
            ideal_col = info["ideal_function"]
            delta = np.abs(self.train_df[train_col] - self.ideal_df[ideal_col])
            self.max_devs[ideal_col] = delta.max() * np.sqrt(2)

    def map_test_data(self):
        """
        Compares each test point to the 4 selected ideal functions.

        If the absolute difference (delta_y) between a test point and an ideal function
        is within the allowed threshold, it is assigned and saved.

        Returns:
            pd.DataFrame: A DataFrame with columns [x, y, delta_y, ideal_function]
        """
        self.compute_thresholds()
        mapped = []

        for _, row in self.test_df.iterrows():
            x_test, y_test = row["x"], row["y"]
            match_found = False

            for train_col, info in self.matches.items():
                ideal_col = info["ideal_function"]
                y_match = self.ideal_df.loc[self.ideal_df["x"] == x_test, ideal_col]

                if y_match.empty:
                    continue  # x not found

                y_ideal_val = y_match.values[0]
                delta_y = abs(y_test - y_ideal_val)

                if delta_y <= self.max_devs[ideal_col]:
                    mapped.append({
                        "x": x_test,
                        "y": y_test,
                        "delta_y": delta_y,
                        "ideal_function": ideal_col
                    })
                    match_found = True
                    break

            if not match_found:
                mapped.append({
                    "x": x_test,
                    "y": y_test,
                    "delta_y": None,
                    "ideal_function": None
                })

        # Convert to DataFrame and save
        mapped_df = pd.DataFrame(mapped)
        mapped_df.to_csv(os.path.join(self.data_dir, "mapped_test_points.csv"), index=False)
        print(f"[INFO] Mapped {mapped_df['ideal_function'].notna().sum()} of {len(mapped_df)} test points.")
        return mapped_df

class Visualizer(FunctionProcessor):
    """
    Uses Bokeh to generate an interactive HTML visualization
    that includes:

    - Training functions
    - Chosen ideal functions
    - Mapped test points (color-coded)
    - Unmatched test points (gray crosses)

    Attributes:
        matches (dict): Mappings from training to ideal functions
        output_path (str): Path to save the HTML visualization
    """

    def __init__(self, matches, data_dir="data", output_path="data/function_mapping_visualization.html"):
        super().__init__(data_dir)
        self.matches = matches
        self.output_path = output_path

    def create_plot(self):
        """
        Creates and saves a Bokeh plot visualizing:
        - Training vs ideal functions
        - Test point mappings
        - Unmatched test points

        Output:
            Saves an HTML file to the path defined in `self.output_path`.
        """
        
        # Load everything
        self.load_all()
        mapped_df = pd.read_csv(os.path.join(self.data_dir, "mapped_test_points.csv"))

        # Prep colors
        palette = Category10[10]
        match_colors = {
            v["ideal_function"]: palette[i]
            for i, v in enumerate(self.matches.values())
        }

        # Set up Bokeh
        output_file(self.output_path)
        p = figure(title="Training vs Ideal Functions with Mapped Test Points",
                   x_axis_label="x", y_axis_label="y", width=900, height=600)

        # Plot training functions
        for i, train_col in enumerate(['y1', 'y2', 'y3', 'y4']):
            p.line(self.train_df['x'], self.train_df[train_col],
                   legend_label=f"Training {train_col}",
                   line_width=2, color=palette[i+4])

        # Plot ideal functions
        for train_col, match in self.matches.items():
            ideal_col = match["ideal_function"]
            color = match_colors[ideal_col]
            p.line(self.ideal_df['x'], self.ideal_df[ideal_col],
                   legend_label=f"Ideal {ideal_col}",
                   line_width=2, line_dash='dashed', color=color)

        # Plot matched test points
        for ideal_col, color in match_colors.items():
            df = mapped_df[mapped_df['ideal_function'] == ideal_col]
            p.scatter(df['x'], df['y'], size=6, color=color,
                      marker='circle', alpha=0.8,
                      legend_label=f"Test → {ideal_col}")

        # Plot unmatched
        unmatched = mapped_df[mapped_df['ideal_function'].isna()]
        if not unmatched.empty:
            p.scatter(unmatched['x'], unmatched['y'], size=8, color='gray',
                      alpha=0.6, marker='cross', legend_label="Unmatched")

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        save(p)
        print(f"[INFO] Visualization saved to: {self.output_path}")