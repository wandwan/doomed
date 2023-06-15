import pandas as pd
import numpy as np

def remove_correlated_columns(df):
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    # Drop correlated columns
    df = df.drop(to_drop, axis=1)
    
    return df
