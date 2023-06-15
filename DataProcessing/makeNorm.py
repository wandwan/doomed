import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def log_normalize_df(df):
    # subtract minimum element from each column and apply log
    df = np.log(df - df.min() + 1)
    
    return df

def normalize_df(df, exclude_cols=[]):
    # get columns to normalize
    cols_to_normalize = [col for col in df.columns if col not in exclude_cols and df[col].dtype == 'float64']
    
    # normalize columns using StandardScaler
    scaler = StandardScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    return df
