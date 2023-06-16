import matplotlib.pyplot as plt
import numpy as np
from splitData import split_data, split_numpy_data
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import re

# Read in data
data_df = pd.read_csv('./data/cleaned_train.csv')

def train(data_df, model_function, n_iterations=100, folds=5, stratify=True):
    # Use only 100 rows for testing
    data_df = data_df[:300]

    # Split data into train and validation sets
    folds = split_data(data_df)
    for i in range(len(folds)):
        X_val_df: pd.DataFrame = folds[i][0]
        Y_val_df: pd.DataFrame = folds[i][1]
        X_train_df: pd.DataFrame = pd.concat([folds[j][0] for j in range(len(folds)) if j != i])
        Y_train_df: pd.DataFrame = pd.concat([folds[j][1] for j in range(len(folds)) if j != i])
        
        
        X_train = X_train_df.to_numpy(dtype=np.float32)
        X_val = X_val_df.to_numpy(dtype=np.float32)
        Y_train = np.round(Y_train_df.to_numpy()).astype(int)
        Y_val = np.round(Y_val_df.to_numpy()).astype(int)
        train_loss, val_loss, y_pred = model_function(X_train, Y_train, X_val, Y_val, n_iterations)

        # Calculate accuracy
        accuracy = accuracy_score(Y_val, y_pred)
        print('Accuracy:', accuracy)

        # Graph train and validation loss
        plt.title('Train and Validation Loss for fold ' + str(i))
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.show()
    
def XGBoost(X_train, Y_train, X_val, Y_val, num_iterations):
    from DataProcessing.generatePairs import generate_pairs

    # Generate pairs
    X_train, Y_train = generate_pairs(X_train, Y_train)
    X_val, Y_val = generate_pairs(X_val, Y_val)

    # Train XGBoost model
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.005, 
        'max_depth': 4,
        # 'colsample_bytree': 0.50,
        # 'subsample': 0.80,
        'eta': 0.03,
        'gamma': 1.5,
        # 'lambda': 70,
        # 'min_child_weight': 8,
        # 'eval_metric':'logloss',
        # 'tree_method': 'gpu_hist',
        # 'predictor':'gpu_predictor',
        'random_state': 423847893,
    }

    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dval = xgb.DMatrix(X_val, label=Y_val)
    res = {}

    evallist = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(xgb_params, dtrain, num_boost_round=num_iterations, evals=evallist, early_stopping_rounds=30, evals_result=res)

    # Make predictions on validation set
    y_pred = model.predict(dval)
    y_pred = np.round(y_pred)
    return res['train']['logloss'], res['validation']['logloss'], y_pred
