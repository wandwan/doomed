import numpy as np
from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.2, random_state=42, stratify=True):
    """
    Splits a dataframe into train and validation sets.

    Parameters:
    data (dataframe): The data to be split.
    test_size (float): The proportion of the data to be used for validation.
    random_state (int): The random seed to be used for shuffling the data.
    stratify (bool): Whether or not to stratify the data based on the target variable.

    Returns:
    X_train (dataframe): The training data.
    X_val (dataframe): The validation data.
    y_train (series): The target variable for the training data.
    y_val (series): The target variable for the validation data.
    """
    y = data["Class"]
    X = data.drop(columns=["Class", "Id"])

    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_val, y_train, y_val

def split_numpy_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Splits numpy arrays into train and validation sets.

    Parameters:
    X (numpy array): The feature data to be split.
    y (numpy array): The target variable to be split.
    test_size (float): The proportion of the data to be used for validation.
    random_state (int): The random seed to be used for shuffling the data.
    stratify (bool): Whether or not to stratify the data based on the target variable.

    Returns:
    X_train (numpy array): The training feature data.
    X_val (numpy array): The validation feature data.
    y_train (numpy array): The target variable for the training data.
    y_val (numpy array): The target variable for the validation data.
    """
    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_val, y_train, y_val