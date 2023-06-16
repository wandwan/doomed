import pandas as pd
from convToNum import convert_to_numeric
from fillNan import fill_na_with_kmeans
from makeInt import convert_to_integer
from makeNorm import normalize_df
from featureCombos import combo_features
from correlatedProcess import remove_correlated_columns
from generatePairs import generate_pairs

input_file = "./data/train.csv"
output_file = "./data/cleaned_train.csv"
selected = ['AB/DA', 'BP*DU', 'DF*GL_2',
       'DI/DL', 'CD/GE', 'EE*EP_2', 'AZ/EP', 'CD/FD', 'FR/GE', 'AF/DL',
       'DA_2*GF', 'AB/DE', 'DU*GH', 'BQ/GF', 'DF*DI_2', 'FC/FL', 'CC/DI',
       'DU*FL_2', 'AF/EG', 'CC_2*DN', 'AF/AY', 'BN/DI', 'DI_2*DU', 'DF*GL',
       'DL_2*EE', 'AB_2*BQ', 'DI*DU', 'BQ*DY', 'CR_2*DH', 'BQ*DY_2', 'AF/EP',
       'AB/EG', 'AB*EL', 'AY/DU', 'CC/DU', 'CR/DU', 'DA_2*EE', 'AB/FD',
       'DA_2*DE', 'DU/GL', 'DL/DY', 'CD/EE', 'BQ_2*FE', 'AF/GE', 'AB/CR',
       'DU/FI', 'BC*FL_2', 'CC*CR_2', 'AB/DU', 'AB/FL', 'BC*DU_2', 'DU_2*FR',
       'DH/DU', 'DA/DI', 'AB/CH', 'BQ/EE', 'AM/DU']

# Read input file into a dataframe
df = pd.read_csv(input_file)

# Convert all columns to numbers
df = convert_to_numeric(df)

# Fill NaN values using kmeans
df = fill_na_with_kmeans(df)

# Convert all columns to integer
df = convert_to_integer(df)

# Normalize the data
df = normalize_df(df, ['Class', 'Id', 'EJ'])

# Remove whitespace from column names
df.columns = df.columns.str.replace(' ', '')

# Remove correlated columns
df = remove_correlated_columns(df)

# Create new features
df = combo_features(df, selected)

print(df.head())

# Save cleaned dataframe to output file
df.to_csv(output_file, index=False)
