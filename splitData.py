import numpy as np
from sklearn.model_selection import StratifiedKFold

def split_data(data, n_folds=5, random_state=42):
    ### Split data into n_folds folds
    features = data.drop("Class", axis=1)
    targets = data["Class"]
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    folds = []
    for train_index, test_index in skf.split(features, targets):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]
        folds.append((X_train, y_train))
    return folds

def split_numpy_data(X, y, n_folds=5, random_state=42, stratify=True):
    ### Split data into n_folds folds
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    folds = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        folds.append((X_train, y_train))
    return folds