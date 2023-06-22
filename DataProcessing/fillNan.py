import pandas as pd
from sklearn.impute import KNNImputer

def fill_na_with_kmeans(df):
    # Drop the Id and Class columns
    id_col = df['Id']
    class_col = df['Class']
    df = df.drop(['Id', 'Class'], axis=1)

    # Create a KNNImputer object with k=5
    imputer = KNNImputer(n_neighbors=5)

    # Impute the NaNs in the dataframe using the KNNImputer
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Add back the Id and Class columns
    df_imputed.insert(0, 'Id', id_col)
    df_imputed.insert(len(df_imputed.columns), 'Class', class_col)

    return df_imputed
    