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

def logAndScaleColumns(df, exclude_cols=[]):
        # Get columns to scale
        data = df.copy()
        data: pd.DataFrame = data.drop(exclude_cols, axis=1)
        
        # Make all values positive broadcast subtraction to all rows
        for col in data.columns:
            data[col] = data[col] - data[col].min() + 1
        
        # Take log of each column
        log_data = np.log(data)
        
        # Scale each column to mean 0
        scaler = StandardScaler(with_std=False)
        scaled_data = scaler.fit_transform(log_data)
        
        # Convert scaled data back to dataframe
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
        
        # add back the excluded columns
        scaled_data = pd.concat([scaled_data, df[exclude_cols]], axis=1)
        
        # Return scaled data
        return scaled_data