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

def tree_clean(input_file):
    selected = ["DU_l/FI", "DU_l/EP", "DU_l*AR", "DU_l/CC", "EH_l/GL", "FE/DA_2", "FD_l/GL", "BC_l/DE", "EB_l/DA", "CD/CR", "AR/DE", "AB/DE", "FE_l/DA"]

    # Read input file into a dataframe
    df = pd.read_csv(input_file)

    # Convert all columns to numbers
    df = convert_to_numeric(df)

    # Fill NaN values using kmeans
    df = fill_na_with_kmeans(df)

    # Convert all columns to integer
    df = convert_to_integer(df)

    # Normalize the data
    # df = normalize_df(df, ['Class', 'Id', 'EJ'])

    # Remove whitespace from column names
    df.columns = df.columns.str.replace(' ', '')

    # Remove correlated columns
    df = remove_correlated_columns(df)

    # Create new features
    df = combo_features(df, selected)
    
    return df.copy()

tree_clean(input_file).to_csv(output_file, index=False)