import matplotlib.pyplot as plt
import numpy as np
from splitData import split_data
import pandas as pd
import xgboost as xgb
from typing import Callable
import re

# Read in data
data_df = pd.read_csv('./data/cleaned_train.csv')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from splitData import split_data

def _nFoldCV(df, data_processing_function, model_function, score_function, n_iterations=1000000, n_folds=5, stratify=True, plot_loss_curve=False):
    # scores holds floats of scores for each fold
    scores = []
    # losses holds tuples of (train_loss, val_loss)
    losses = []
    # models holds strings of model paths or model objects
    models = []
    
    for i in range(n_folds):
        for X_train_df, X_val_df, Y_train_df, Y_val_df in split_data(df, n_folds=n_folds, stratify=stratify):
            # Preprocess data
            X_train, Y_train = data_processing_function(X_train_df, Y_train_df)
            X_val, Y_val = data_processing_function(X_val_df, Y_val_df)
            
            # Target's should be 0-1
            Y_train = np.round(Y_train_df.to_numpy()).astype(int)
            Y_val = np.round(Y_val_df.to_numpy()).astype(int)
            
            # Train model
            train_loss, val_loss, y_pred, model = model_function(X_train, Y_train, X_val, Y_val, n_iterations)
            
            # Save model, score, and and minimum val_loss
            models.append(model)
            if score_function is not None:
                score = score_function(Y_val, y_pred)
                scores.append(score)
            min_idx = np.argmin(np.array(val_loss))
            losses.append((train_loss[min_idx], val_loss[min_idx]))
            
            # Plot train and validation loss curves
            if plot_loss_curve:
                plt.plot(train_loss, label='Train Loss')
                plt.plot(val_loss, label='Validation Loss')
                plt.title(f'Fold {i} Loss Curve')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.legend()
                plt.show(block=False)
    return models, scores, losses

def train(df : pd.DataFrame, data_processing_function : Callable, 
          model_function : Callable, score_function:(Callable|None)=None, 
          inference_function:(Callable|None) = None, n_iterations:int=1000000, 
          n_folds:int=5, stratify:bool=True, leave_one_out:bool=False, plot_loss_curve:bool=False) -> list:
    """
    Function to train a model using n-fold cross validation and return the best model from each fold.
    Complete training pipeline invariant to the type of model used as long as the model function, data processing function, and score function are provided.
    Supports Leave One Out testing, where for every row in the dataframe, the model is trained on all other rows and tested on that row.
    If Leave One Out testing is used, the inference and score functions must be provided.

    Args:
        df (DataFrame): Dataframe containing the data to train and possibly test on (if leave_one_out is True)
        data_processing_function (function): A function that must take in 2 DataFrames (X_before and Y_before) 
                                             as arguments and returns 2 new DataFrames (X_type_ and Y) where X 
                                             is the features and Y is the target (X must not contain the target)
        model_function (function): The main function for actually training a specific model, must take in 5 arguments 
                                   (X_train, Y_train, X_val, Y_val, n_iterations) 4 DataFrames and an int,
                                   and return 4 values train_loss, val_loss, y_pred, model. 
                                   train_loss should be a list of floats of the training losses
                                   val_loss should be a list of floats of the validation losses
                                   y_pred should be a list of floats of the predictions on the validation set
                                   model should be the trained model (as either a string or the actual model object)
        score_function (function, optional): A function that takes in 2 arguments (Y_true, Y_pred) both of type np arrays and returns a float.
                                             This function is used to score the model on the validation set. Defaults to None.
        inference_function (function, optional): A function that takes in 2 arguments (X, model) a dataframe and a model string or object
                                                 (based on what your model_function returns) and returns an np array of floats.
                                                 This function is used to make predictions on the test set. Defaults to None.
        n_iterations (int, optional): Maximum number of iterations to train on. Defaults to 1000000.
        n_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        stratify (bool, optional): whether or not to use stratified K-Fold or regular K-Fold. Defaults to True.
        leave_one_out (bool, optional): Whether or not to do Leave-one-out testing, if this is set to true, 
                                        you must pass in an inference and score function. Defaults to False.
        plot_loss_curve (bool, optional): whether or not to plot a loss curve for each iteration of CV, 
                                          does not work with leave-one-out testing. Defaults to False.

    Raises:
        ValueError: If leave_one_out is True and inference_function or score_function is None
                    (since we need to make predictions on the test set and score them)

    Returns:
        list: A list of models from each fold
    """
    
    
    models = []
    scores = []
    losses = []
    
    # Leave one out testing (for every data point, train on all other data points and test on that data point)
    if leave_one_out:
        if inference_function is None or score_function is None:
            raise ValueError('Inference and score functions must be provided for leave one out testing')
        
        loo_scores = []
        for i in range(len(df)):
            # Leave out one data point
            X_test_df = df.iloc[[i]]
            Y_test_df = X_test_df.pop('Class')
            X_df = df.drop(i)
            
            # Overwrite models scores and losses every time since we only leave out one data point
            # so the models should be about the same
            models, scores, losses = _nFoldCV(X_df, data_processing_function, model_function, score_function, n_iterations, n_folds, stratify)
            X_test, Y_test = data_processing_function(X_test_df, Y_test_df)
            val_loss = [loss[1] for loss in losses]
            best_model_index = val_loss.index(min(val_loss))
            best_model = models[best_model_index]
            y_pred = inference_function(X_test, best_model)
            score = score_function(Y_test, y_pred)
            loo_scores.append(score)
        
        # Plot scores
        plt.bar(range(len(df)), loo_scores)
        plt.title('Leave One Out Scores')
        plt.xlabel('Data Point')
        plt.ylabel('Score')
        plt.show()
        print('Average LOO Score:', np.mean(loo_scores))
        print('Median LOO Score:', np.median(loo_scores))
    else:
        models, scores, losses = _nFoldCV(df, data_processing_function, model_function, score_function, n_iterations, n_folds, stratify, plot_loss_curve)
        plt.bar(range(len(scores)), scores)
        plt.title('Validation Scores')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.show()
    val_loss = [loss[1] for loss in losses]
    train_loss = [loss[0] for loss in losses]
    
    # Plot scores and losses
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    sorted_scores = sorted(scores, reverse=True)
    sorted_val_loss = sorted(val_loss)
    sorted_train_loss = sorted(train_loss)
    axs[0].bar(range(len(sorted_scores)), sorted_scores)
    axs[0].set_title('Validation Scores')
    axs[0].set_xlabel('Fold')
    axs[0].set_ylabel('Score')
    axs[1].bar(range(len(sorted_val_loss)), sorted_val_loss)
    axs[1].set_title('Validation Losses')
    axs[1].set_xlabel('Fold')
    axs[1].set_ylabel('Loss')
    axs[2].bar(range(len(sorted_train_loss)), sorted_train_loss)
    axs[2].set_title('Train Losses')
    axs[2].set_xlabel('Fold')
    axs[2].set_ylabel('Loss')
    plt.tight_layout()
    plt.show()
    
    # Print scores and losses
    print('Average Validation Score:', np.mean(scores))
    print('Median Validation Score:', np.median(scores))
    print('Average Validation Loss:', np.mean(val_loss))
    print('Median Validation Loss:', np.median(val_loss))
    print('Average Training Loss:', np.mean(train_loss))
    print('Median Training Loss:', np.median(train_loss))
    
    # Return models
    return models
    
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
            'eval_metric': 'logloss',
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