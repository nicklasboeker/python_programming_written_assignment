import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, Integer, MetaData, Table
from sqlalchemy.orm import declarative_base, Session
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category10

"""STEP 1: Set Up SQLite Database
1. Create database
2. Use SQLAlchemy to define two tables: training_data, ideal_functions
3. Insert all data into tables"""

# Load CSV files
train_df = pd.read_csv("train.csv")
ideal_df = pd.read_csv("ideal.csv")

# Create SQLite database
engine = create_engine("sqlite:///functions.db")
Base = declarative_base()

# Define training_data table
class TrainingData(Base):
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)

# Define ideal_functions table dynamically
def create_ideal_function_table(metadata):
    columns = [Column('id', Integer, primary_key=True), Column('x', Float)]
    columns += [Column(f'y{i}', Float) for i in range(1, 51)]
    return Table('ideal_functions', metadata, *columns)

# Create tables
metadata = MetaData()
ideal_functions_table = create_ideal_function_table(metadata)
Base.metadata.create_all(engine)
metadata.create_all(engine)

# Insert training data
with Session(engine) as session:
    for _, row in train_df.iterrows():
        session.add(TrainingData(x=row['x'], y1=row['y1'], y2=row['y2'], y3=row['y3'], y4=row['y4']))
    session.commit()

# Insert ideal functions data
with engine.connect() as conn:
    ideal_df.insert(0, 'id', range(1, len(ideal_df) + 1))  # add ID column
    ideal_df.to_sql('ideal_functions', conn, if_exists='replace', index=False)


"""STEP 2: Find the 4 Best-Fitting Ideal Functions:
1. Load both datasets into Pandas again (here we use them directly without SQL for performance).
2. Loop through all 4 training functions (y1–y4).
3. For each one, calculate the mean squared error (MSE) against each of the 50 ideal functions (y1–y50).
4. Pick the ideal function with the lowest MSE that hasn't already been chosen."""

# Load csvs again
train_df = pd.read_csv("train.csv")
ideal_df = pd.read_csv("ideal.csv")

# Keep track of matached ideal functions
chosen_matches = {}
already_used = set()

# Loop over the 4 training functions
for train_col in ['y1', 'y2', 'y3', 'y4']:
    min_error = float('inf')
    best_match = None

    # Check against all ideal functions
    for ideal_col in [f'y{i}' for i in range(1, 51)]:
        if ideal_col in already_used:
            continue # only one match per ideal function

        # Calculate MSE
        error = np.mean((train_df[train_col] - ideal_df[ideal_col]) ** 2)

        # Save the best match
        if error < min_error:
            min_error = error
            best_match = ideal_col

    # Save the match
    chosen_matches[train_col] = {"ideal_function": best_match, "error": min_error}
    already_used.add(best_match)

# Print the results
for train_func, result in chosen_matches.items():
    print(f"{train_func} -> {result['ideal_function']} (MSE: {result['error']:.6f})")


"""STEP 3: Mapping Test Data Points to the 4 Chosen Ideal Functions
1. Compute max deviations
2. Load test data & try to match
3. Save results"""

# Map of training function to ideal function (from previous result)
chosen_matches = {'y1':'y42','y2': 'y41','y3': 'y11','y4': 'y48'}

# Compute max deviations for each match
max_deviations = {}
for train_col, ideal_col in chosen_matches.items():
    delta = np.abs(train_df[train_col] - ideal_df[ideal_col])
    max_dev = delta.max()
    max_deviations[ideal_col] = max_dev * np.sqrt(2)

# Load test data
test_df = pd.read_csv("test.csv")

# Prepare result list
mapped_points = []

# Check each test (x,y) point
for _, row in test_df.iterrows():
    x_test, y_test = row['x'], row['y']
    match_found = False

    # Try matching to each ideal function
    for train_col, ideal_col in chosen_matches.items():
        # Get corresponding y_ideal for x
        y_ideal_at_x = ideal_df.loc[ideal_df['x'] == x_test, ideal_col]

        if y_ideal_at_x.empty:
            continue # x not found

        y_ideal_val = y_ideal_at_x.values[0]
        delta_y = abs(y_test - y_ideal_val)

        # Check against threshold
        if delta_y <= max_deviations[ideal_col]:
            mapped_points.append({'x': x_test, 'y': y_test, 'delta_y': delta_y, 'ideal_function': ideal_col})
            match_found = True
            break # only assign to one function

        if not match_found:
            # Optionally log unmapped point
            mapped_points.append({'x': x_test,'y': y_test,'delta_y': None,'ideal_function': None})

# Save Results
mapped_df = pd.DataFrame(mapped_points)
mapped_df.to_csv("mapped_test_points.csv", index=False)


"""STEP 4: Save Test Mappings to SQLite"""

# Load the mapped results
mapped_df = pd.read_csv("mapped_test_points.csv")
#Connect to database
engine = create_engine("sqlite:///functions.db")
# Save results to new table
mapped_df.to_sql("mapped_test_data", engine, if_exists="replace", index=False)


"""STEP 5: Visualize the Results with Bokeh
1. Training data: 4 coloured lines
2. Chosen Ideal Functions: 4 corresponding ideal lines
3. Test Points:
- Color-coded by which ideal function they matched (y42, y41, y11, y48)
- Unmatched test points (if any) shown in gray or X markers"""

# Load all needed data
train_df = pd.read_csv("train.csv")
ideal_df = pd.read_csv("ideal.csv")
mapped_df = pd.read_csv("mapped_test_points.csv")

# Set up Bokeh output
output_file("function_mapping_visualization.html")

# Create figure
p = figure(title = "Training vs. Ideal Functions with Mapped Test Points", x_axis_label='x', y_axis_label='y', width=900, height=600)

# Color palette
colors = Category10[10]
match_colors = {'y42': colors[0], 'y41': colors[1], 'y11': colors[2], 'y48': colors[3]}

# Plot training functions
for i, col in enumerate (['y1', 'y2', 'y3', 'y4']):
    p.line(train_df['x'], train_df[col], legend_label=f"Training {col}",line_width=2, color=colors[i+4], line_dash='solid')

# Plot corresponding ideal functions
match_map = {'y1': 'y42', 'y2': 'y41', 'y3': 'y11', 'y4': 'y48'}
for train_col, ideal_col in match_map.items():
    p.line(ideal_df['x'], ideal_df[ideal_col],legend_label=f"Ideal {ideal_col}", line_width=2,color=match_colors[ideal_col], line_dash='dashed')

# Plot matched test points
for ideal_col, color in match_colors.items():
    df = mapped_df[mapped_df['ideal_function'] == ideal_col]
    p.scatter(df['x'], df['y'], size=6, color=color,legend_label=f"Test points → {ideal_col}", alpha=0.8,marker="circle")

# Plot unmatched points
unmatched = mapped_df[mapped_df['ideal_function'].isna()]
if not unmatched.empty:
    p.scatter(unmatched['x'], unmatched['y'], size=8, color="gray",alpha=0.6, legend_label="Unmatched", marker="cross")

# Show legend and plot

p.legend.location = "top_left"
p.legend.click_policy = "hide"
show(p)