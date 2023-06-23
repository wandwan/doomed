import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import KFold

def split_data(data, n_folds=5, random_state=42, stratify=True, stratify_by_alpha=True):
    ### Split data into n_folds folds
    features = data.drop("Class", axis=1)
    
    ### Stratify by alpha
    if(stratify_by_alpha):
        df = pd.read_csv("data/greeks.csv")
        targets = df["Alpha"]
    else:
        targets = data["Class"]
        
    if stratify:
        skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    else:
        skf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    
    # Return generator for train and validation sets
    for train_index, test_index in skf.split(features, targets):
        X_train, X_val = features.iloc[train_index], features.iloc[test_index]
        if stratify_by_alpha:
            targets = data["Class"]
        y_train, y_val = targets.iloc[train_index], targets.iloc[test_index]
        
        yield X_train, X_val, y_train, y_val