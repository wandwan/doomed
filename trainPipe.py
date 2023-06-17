import matplotlib.pyplot as plt
import numpy as np
from splitData import split_data, split_numpy_data
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import re

# Read in data
data_df = pd.read_csv('./data/cleaned_train.csv')

def train(data_df, model_function, n_iterations=100000, folds=5, stratify=True):
    # Use only 100 rows for testing
    # data_df = data_df[:300]
    i = 0
    # Split data into train and validation sets
    for X_train_df, X_val_df, Y_train_df, Y_val_df in split_data(data_df, n_folds=folds):
        
        X_train = X_train_df.to_numpy(dtype=np.float32)
        X_val = X_val_df.to_numpy(dtype=np.float32)
        Y_train = np.round(Y_train_df.to_numpy()).astype(int)
        Y_val = np.round(Y_val_df.to_numpy()).astype(int)
        print(X_train.shape, X_val.shape, Y_val.shape, Y_val.shape)
        train_loss, val_loss, y_pred = model_function(X_train, Y_train, X_val, Y_val, n_iterations)

        # Calculate accuracy
        # accuracy = accuracy_score(Y_val, y_pred)
        # print('Accuracy:', accuracy)

        # Graph train and validation loss
        # plt.title('Train and Validation Loss for fold ' + str(i))
        # i += 1
        # plt.plot(train_loss, label='Train Loss')
        # plt.plot(val_loss, label='Validation Loss')
        # plt.legend()
        # plt.show()
    
def XGBoost(X_train, Y_train, X_val, Y_val, num_iterations):
    from DataProcessing.generatePairs import generate_pairs

    # Generate pairs
    X_train, Y_train = generate_pairs(X_train, Y_train)
    X_val, Y_val = generate_pairs(X_val, Y_val)
    # Shuffle X_train and Y_train together
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    # Train XGBoost model
    xgb_params = {
            'learning_rate': 0.03,
            'max_depth': 7,
            'lambda': 1.3,
            'alpha':.2,
            'colsample_bytree':.4,
            'grow_policy': 'lossguide',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            
            'verbosity': 0,
            'random_state': 1234123579,
        }

    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dval = xgb.DMatrix(X_val, label=Y_val)
    res = {}

    evallist = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(xgb_params, dtrain, num_boost_round=num_iterations, evals=evallist, early_stopping_rounds=100, evals_result=res)

    # Make predictions on validation set
    y_pred = model.predict(dval)
    y_pred = np.round(y_pred)
    return res['train']['logloss'], res['validation']['logloss'], y_pred
train(data_df, XGBoost)