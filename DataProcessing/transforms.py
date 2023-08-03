import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, PowerTransformer, QuantileTransformer


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    ej_map = {'A': 0, 'B': 1}
    id_map = {id_val: i+1 for i, id_val in enumerate(df['Id'].unique())}
    df['EJ'] = df['EJ'].map(ej_map)
    df['Id'] = (df['Id'].map(id_map)).astype(int)
    df['Class'] = df['Class'].astype(int)
    return df

def remove_correlated_columns(df:pd.DataFrame, threshhold:float=0.95) -> pd.DataFrame:
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] >= threshhold)]
    
    # replace correlated columns with their mean
    for column in to_drop:
        mean = df[column].mean()
        df.drop(column, axis=1, inplace=True)
        df[column] = mean
    
    return df

# Transform the data using PCA. Use test data to calculate components if available, but always fit on train data.
# n_components can be a float between 0 and 1, or an int. If it is a float, it will be treated as the minimum
# verbose will plot the explained variance ratios and save them to pca.png
def pca(df: pd.DataFrame, x_test: pd.DataFrame | None=None, n_components: float=0.95, verbose: bool=False) -> pd.DataFrame:
    pca_val:PCA = PCA(n_components=n_components)
    id_col = df['Id']
    class_col = df['Class']
    df = df.drop(["Class", "Id"], axis=1)
    
    if x_test is not None:
        pca_val.fit(pd.concat([df, x_test], axis=1))
    else:
        pca_val.fit(df)
    
    if verbose:
        plt.plot(np.cumsum(pca_val.explained_variance_ratio_))
        print("Explained variance ratios: ", np.cumsum(pca_val.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.savefig("pca.png")
        plt.plot(np.cumsum(pca_val.explained_variance_ratio_))
        print("Explained variance ratios: ", np.cumsum(pca_val.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.savefig("pca.png")
    
    transformed = pca_val.transform(df.to_numpy())
    df = pd.DataFrame(transformed, columns=[f"PC{i}" for i in range(transformed.shape[1])])
    return pd.concat([df, id_col, class_col], axis=1)

def convert_to_integer(df: pd.DataFrame) -> pd.DataFrame:
    # In the competition data, there are some columns that are integers, but are 
    # obfuscated as floats. This function will convert those columns to integers.

    int_denominators = {
     'AB': 0.004273,
     'AF': 0.00242,
     'AH': 0.008709,
     'AM': 0.003097,
     'AR': 0.005244,
     'AX': 0.008859,
     'AY': 0.000609,
     'AZ': 0.006302,
     'BC': 0.007028,
     'BD': 0.00799,
     'BN': 0.3531,
     'BP': 0.004239,
     'BQ': 0.002605,
     'BR': 0.006049,
     'BZ': 0.004267,
     'CB': 0.009191,
     'CC': 6.12e-06,
     'CD': 0.007928,
     'CF': 0.003041,
     'CH': 0.000398,
     'CL': 0.006365,
     'CR': 7.5e-05,
     'CS': 0.003487,
     'CU': 0.005517,
     'CW': 9.2e-05,
     'DA': 0.00388,
     'DE': 0.004435,
     'DF': 0.000351,
     'DH': 0.002733,
     'DI': 0.003765,
     'DL': 0.00212,
     'DN': 0.003412,
     'DU': 0.0013794,
     'DV': 0.00259,
     'DY': 0.004492,
     'EB': 0.007068,
     'EE': 0.004031,
     'EG': 0.006025,
     'EH': 0.006084,
     'EL': 0.000429,
     'EP': 0.009269,
     'EU': 0.005064,
     'FC': 0.005712,
     'FD': 0.005937,
     'FE': 0.007486,
     'FI': 0.005513,
     'FR': 0.00058,
     'FS': 0.006773,
     'GB': 0.009302,
     'GE': 0.004417,
     'GF': 0.004374,
     'GH': 0.003721,
     'GI': 0.002572
    }

    for k, v in int_denominators.items():
        df[k] = np.round(df[k]/v,1).astype(int)
        
    return df

# Possible strategies: 'KNN', 'mean', 'median', '0', '1', 'most_frequent'
def fill_nans(df, strategy='KNN'):
    # Drop the Id and Class columns
    id_col, class_col = df['Id'], df['Class']
    df = df.drop(['Id', 'Class'], axis=1)

    imputer = None
    # Create a KNNImputer object with k=5
    match strategy:
        case 'KNN':
            imputer = KNNImputer(n_neighbors=5)
        case 'mean':
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        case 'median':
            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        case '0':
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        case '1':
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=1)
        case 'most_frequent':
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        case _:
            raise ValueError(f"Invalid strategy: {strategy}")

    # Impute the NaNs in the dataframe using the KNNImputer
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns) # type: ignore

    # Add back the Id and Class columns
    df_imputed.insert(0, 'Id', id_col)
    df_imputed.insert(len(df_imputed.columns), 'Class', class_col)

    return df_imputed

def normalize(df: pd.DataFrame, cols_to_skip: list[str]=[], strategy: str = "StandardScaler", log: bool = False) -> pd.DataFrame:
    cols_to_skip += ["Id", "Class"]
    originals = df[cols_to_skip]
    df = df.drop(cols_to_skip, axis=1)
    
    if log:
        df = pd.DataFrame(np.log1p(df), columns=df.columns)
    
    match strategy:
        case "StandardScaler":
            scaler = StandardScaler()
        case "MinMaxScaler" | "MinMax":
            scaler = MinMaxScaler()
        case "PowerTransformer" | "box-cox":
            scaler = PowerTransformer('box-cox')
        case "QuantileTransformer":
            scaler = QuantileTransformer()
        case _:
            raise ValueError(f"Invalid strategy: {strategy}")
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return pd.concat([df, originals], axis=1)
    
    