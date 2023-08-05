from DataProcessing.generatePairs import generate_pairs, generate_pairs_for_validation
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from splitData import split_data

def leaveOneOut(df: pd.DataFrame, model, criterion):
    """
    Leave one out cross validation
    """
    for index in range(len(df)):
        # Get the row to leave out
        row = df.iloc[index]
        # Drop the row from the dataframe
        df = df.drop(index)
        # Get the target
        target = row['Class']
        # Remove the target from the dataframe
        df = df.drop('Class', axis=1)
        # Train the model
        model.fit(df.to_numpy(), target.to_numpy())
        # Get the prediction
        pred = model.predict(row.drop('Class').to_numpy().reshape(1, -1))
        # Get the loss
        loss = criterion(pred, row['Class'])
        # Return the loss
        return loss
    
def kFoldCV(df: pd.DataFrame, model, criterion, k=5):
    """
    k-fold cross validation
    """
    losses = []
    for X_train, X_val, y_train, y_val in split_data(df, k):
        # Train the model
        model.fit(X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy())
        # Get the prediction
        pred = model.predict(X_val.to_numpy())
        # Get the loss
        loss = criterion(pred, y_val)
        # Return the loss
        losses.append(loss)
    return np.mean(losses), np.std(losses)

class contrastiveXGBoost:
    def fit(self, X_train, Y_train, X_val=None, Y_val=None, num_iterations=2000, verbose=True, params = None):
        if X_val is None or Y_val is None:
            # Split data into train and validation sets
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)
        
        # Save train sets
        self.X_train = X_train
        self.Y_train = Y_train
        
        # Generate pairs
        X_train, Y_train = generate_pairs(X_train, Y_train, add_noise=False)
        X_val, Y_val = generate_pairs_for_validation(X_train, X_val, Y_train, Y_val, add_noise=False)
        # Shuffle X_train and Y_train together
        perm = np.random.RandomState(seed=42).permutation(len(X_train))
        X_train = X_train[perm]
        Y_train = Y_train[perm]
        
        # Train XGBoost model
        xgb_params = {'learning_rate': 0.018168061411573882, 
                      'min_split_loss': 0.5315667826163368, 
                      'subsample': 0.5657155206937886, 
                      'colsample_bytree': 0.18149665124735848, 
                      'max_depth': 8, 
                      'reg_alpha': 0.8153986720516078, 
                      'reg_lambda': 4.455946009715993,
                      'grow_policy': 'lossguide',
                      'n_jobs': 24,
                      'objective': 'binary:logistic',
                      'eval_metric': 'logloss',
                      'verbosity': 0,
                      'random_state': 423}
        if params is not None:
            xgb_params = params

        dtrain = xgb.DMatrix(X_train, label=Y_train)
        dval = xgb.DMatrix(X_val, label=Y_val)
        res = {}
        evallist = [(dtrain, 'train'), (dval, 'validation')]

        self.model = xgb.train(xgb_params, dtrain, num_boost_round=num_iterations, evals=evallist, early_stopping_rounds=100, evals_result=res, verbose_eval=50 if verbose else False)
    
    def predict(self, X_test):
        # Generate pairs
        Y_test = np.zeros(len(X_test))
        X_test, Y_test = generate_pairs_for_validation(self.X_train, X_test, self.Y_train, Y_test)
        n = len(self.X_train)
        # Predict
        dtest = xgb.DMatrix(X_test)
        preds = self.model.predict(dtest)
        
        final_preds = []
        # Get average prediction
        for i in range(len(X_test)):
            avg = [0, 0]
            elems = [0,0]
            for j in range(i * n, (i + 1) * n):
                if Y_test[j % n] == 1:
                    avg[1] += preds[j]
                    elems[1] += 1
                else:
                    avg[0] += preds[j]
                    elems[0] += 1
            
            final_preds.append((avg[1] / elems[1] + (1 - avg[0] / elems[0])) / 2)
        return np.array(final_preds)
            
    