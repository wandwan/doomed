import pandas as pd
from transforms import *
from featureCombos import calculateNewFeatures

input_file = "./data/train.csv"
output_file = "./data/cleaned_train.csv"

def tree_clean(input_file):
    cols = ['AH', 'CC', 'GI', 'BQ', 'GH', 'DE', 'BR', 'EP', 'EL', 'CL', 'CD', 'AM', 'EU', 'FI', 'EE', 'CU', 'DU', 'GL', 'DL', 'DV']
    selected = ["DU_l/FI", "DU_l/EP", "DU_l*AR", "DU_l/CC", "EH_l/GL", "FE/DA_2", "FD_l/GL", "BC_l/DE", "EB_l/DA", "CD/CR", "AR/DE", "AB/DE", "FE_l/DA"]
    
    # Read input file into a dataframe
    df = pd.read_csv(input_file)
    
    # Remove whitespace from column names
    df.columns = df.columns.str.replace(' ', '')

    # Convert all columns to numbers
    df = convert_to_numeric(df)

    # Fill NaN values using kmeans
    df = fill_nans(df)
    
    # calculate new features
    df2 = calculateNewFeatures(df, 100)
    
    # Convert all columns to integer
    df1 = convert_to_integer(df)
    
    # Select useful columns:
    df1 = df1[cols]
    
    df = pd.concat([df1, df2, df["Class"], df["Id"]], axis=1)

    # Remove correlated columns
    df = remove_correlated_columns(df)
    
    return df.copy()

def pca_clean(input_file):
    # Read input file into a dataframe
    df = pd.read_csv(input_file)
    
    # Remove whitespace from column names
    df.columns = df.columns.str.replace(' ', '')

    # Convert all columns to numbers
    df = convert_to_numeric(df)

    # Fill NaN values using kmeans
    df = fill_nans(df)
    
    # calculate new features
    df = calculateNewFeatures(df, 100)
    
    # Box-Cox transform
    df = normalize(df, ["EJ"], "box-cox")
    
    # PCA
    df = pca(df, n_components=0.98)
    
    return df.copy()

tree_clean(input_file).to_csv(output_file, index=False)
pca_clean(input_file).to_csv("data/pca.csv", index=False)