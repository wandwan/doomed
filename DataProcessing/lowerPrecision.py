import pandas as pd
from convToNum import convert_to_numeric
from fillNan import fill_na_with_kmeans
from makeInt import convert_to_integer

input_file = "./data/train.csv"
output_file = "./data/low_precision.csv"

selected = ["DU_l/FI", "DU_l/EP", "DU_l*AR", "DU_l/CC", "EH_l/GL", "FE/DA_2", "FD_l/GL", "BC_l/DE", "EB_l/DA", "CD/CR", "AR/DE", "AB/DE", "FE_l/DA"]

# Read input file into a dataframe
df = pd.read_csv(input_file)

# Convert all columns to numbers
df = convert_to_numeric(df)

# Fill NaN values using kmeans
df = fill_na_with_kmeans(df)

# Convert all columns to integer
df = convert_to_integer(df)

# Remove whitespace from column names
df.columns = df.columns.str.replace(' ', '')

# Lower the precision of all float columns
float_cols = df.select_dtypes(include=['float'])
for col in float_cols.columns:
    df[col] = df[col].apply(lambda x: '{:.2e}'.format(x))

# Write the output to a fil>e
df.to_csv(output_file, index=False)

