import numpy as np
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import KFold

def split_data(data, n_folds=5, random_state=42, stratify=True):
    ### Split data into n_folds folds
    features = data.drop("Class", axis=1)
    targets = data["Class"]
    if stratify:
        skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    else:
        skf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    
    # Return generator for train and validation sets
    for train_index, test_index in skf.split(features, targets):
        X_train, X_val = features.iloc[train_index], features.iloc[test_index]
        y_train, y_val = targets.iloc[train_index], targets.iloc[test_index]
        yield X_train, X_val, y_train, y_val