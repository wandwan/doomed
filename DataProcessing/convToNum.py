import pandas as pd

def convert_to_numeric(df):
    ej_map = {'A': 0, 'B': 1}
    id_map = {id_val: i+1 for i, id_val in enumerate(df['Id'].unique())}
    df['EJ'] = df['EJ'].map(ej_map)
    df['Id'] = df['Id'].map(id_map).astype('int64')
    return df
    
