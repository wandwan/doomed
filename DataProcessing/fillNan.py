import pandas as pd
from sklearn.impute import KNNImputer

def fill_na_with_kmeans(df):
    # Create a KNNImputer object with k=5
    imputer = KNNImputer(n_neighbors=5)

    # Impute the NaNs in the dataframe using the KNNImputer
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputed
    