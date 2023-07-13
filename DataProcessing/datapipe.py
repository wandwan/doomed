import pandas as pd
from convToNum import convert_to_numeric
from fillNan import fill_na_with_kmeans
from makeInt import convert_to_integer
from makeNorm import normalize_df
from featureCombos import combo_features
from featureCombos import calculateNewFeatures
from correlatedProcess import remove_correlated_columns
from generatePairs import generate_pairs

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
    df = fill_na_with_kmeans(df)
    
    # calculate new features
    df2 = calculateNewFeatures(df, 100, 30)
    
    # Convert all columns to integer
    df1 = convert_to_integer(df)
    
    # Select useful columns:
    df1 = df1[cols]
    
    df = pd.concat([df1, df2, df["Class"], df["Id"]], axis=1)

    
    # Normalize the data
    # df = normalize_df(df, ['Class', 'Id', 'EJ'])


    # Remove correlated columns
    df = remove_correlated_columns(df)

    # Create new features
    # df = combo_features(df, selected)
    
    return df.copy()
def pca_clean(input_file, newFeatures=100, PCAKept=36):
    ### Returns a non-negative dataset that's been log-scaled (then PCA and added with new features).
    # Load data into dataframe
    df = pd.read_csv(input_file)

    # Remove whitespace from column names
    df.columns = df.columns.str.replace(' ', '')
    
    from convToNum import convert_to_numeric
    # Convert all columns to numbers
    df = convert_to_numeric(df)

    # Fill nan's
    from fillNan import fill_na_with_kmeans
    df = fill_na_with_kmeans(df)
    
    arr = calculateNewFeatures(df, newFeatures, PCAKept)
    arr : pd.DataFrame = pd.DataFrame(arr)
    for col in arr.columns:
        arr[col] += -arr[col].min()
    # arr = pd.concat([arr, df["Class"], df["Id"]], axis=1)
    return arr
    

tree_clean(input_file).to_csv(output_file, index=False)
# pca_clean(input_file).to_csv("data/pca.csv", index=False)