import pandas as pd
from scipy.stats import skew
import sys
sys.path.insert(1, '/home/camel/VSC/ICR/DataProcessing')
from DataProcessing.convToNum import convert_to_numeric
from DataProcessing.fillNan import fill_na_with_kmeans

# Load data into dataframe
df = pd.read_csv('data/train.csv')

# Remove whitespace from column names
df.columns = df.columns.str.replace(' ', '')

# Convert all columns to numbers
df = convert_to_numeric(df)

# Fill NaN values using kmeans
df = fill_na_with_kmeans(df)

# Sort columns by number of unique values
unique_counts = df.nunique()
sorted_cols = unique_counts.sort_values().index.tolist()
sorted_df = df[sorted_cols]

# Calculate skewness of each column
skewness = {}
for col in sorted_df.columns:
    skewness[col] = skew(sorted_df[col])

# Print skewness of each column
for col, skew_val in skewness.items():
    print(f"Skewness of column '{col}': {skew_val}")