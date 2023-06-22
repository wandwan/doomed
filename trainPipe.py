import matplotlib.pyplot as plt
import numpy as np
from splitData import split_data
import pandas as pd
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from typing import Callable, Iterable
import sys

def _nFoldCV(df, preprocess, trainModel, score, n_iterations=1000000, n_folds=5, stratify=True, params=None, plot_loss_curve=False):
    # scores holds floats of scores for each fold
    scores = []
    # losses holds tuples of (train_loss, val_loss)
    losses = []
    # models holds strings of model paths or model objects
    models = []
    if preprocess is not None:
        df = preprocess(df)
    df = df.drop("Id", axis=1)
    i = 0
    axs = None
    if plot_loss_curve:
        fig, axs = plt.subplots(n_folds, 1, figsize=(10, 10))
    for X_train, X_val, Y_train, Y_val in split_data(df, n_folds=n_folds, stratify=stratify):
        X_train = X_train.to_numpy(dtype=np.float32)
        X_val = X_val.to_numpy(dtype=np.float32)
        Y_train= np.round(Y_train.to_numpy()).astype(int)
        Y_val = np.round(Y_val.to_numpy()).astype(int)
        
        # Train model
        train_loss, val_loss, y_pred, model = trainModel(X_train, Y_train, X_val, Y_val, n_iterations, params=params)
        
        # Save model, score, and and minimum val_loss
        models.append(model)
        if score is not None:
            score = score(Y_val, y_pred)
            scores.append(score)
        min_idx = np.argmin(np.array(val_loss))
        losses.append((train_loss[min_idx], val_loss[min_idx]))
        
        # Plot train and validation loss curves
        if axs is not None:
            axs[i].plot(train_loss, label='Train Loss')
            axs[i].plot(val_loss, label='Validation Loss')
            axs[i].set_title(f'Fold {i} Loss Curve', loc="right")
            axs[i].set_xlabel('Iteration')
            axs[i].set_ylabel('Loss')
        i += 1    
    if plot_loss_curve:
        plt.tight_layout()
        plt.legend()
        plt.show(block=False)
    return models, scores, losses

def train(df : pd.DataFrame, 
          model : Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int], 
                                     tuple[Iterable[float], Iterable[float], Iterable[float], str|object]],
          preprocess : Callable[[pd.DataFrame], pd.DataFrame] | None = None, 
          score : (Callable[[pd.DataFrame, pd.DataFrame], list[float]] | None)=None, 
          inference:(Callable[[pd.DataFrame, str|object], pd.DataFrame]|None) = None, 
          n_iterations:int=1000000, n_folds:int=5, stratify:bool=True, 
          leave_one_out:bool=False, plot_loss_curve:bool=False, trial:optuna.Trial|None=None) -> list[str|object]:
    """
    Function to train a model using n-fold cross validation and return the best model from each fold.

    Args:
        df (DataFrame): Dataframe containing the data to train on
        model (function): Function to train a specific model. Must take (X_train, Y_train, X_val, Y_val, n_iterations) and return 4 values train_loss, val_loss, y_pred, and model. You can return None for y_pred if you don't use score.
        preprocess (function, optional): Function to preprocess data before training. Must take (X,Y) and return (X,Y)
        score (function, optional): Function to score a prediction. Must take (Y_true, Y_pred) and return score. Defaults to None.
        inference (function, optional): Function for running inference. Must take (X, model) and return predictions. Defaults to None.
        n_iterations (int, optional): Maximum number of iterations to train on. Defaults to 1000000.
        n_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        stratify (bool, optional): Whether or not to use stratified K-Fold over regular K-Fold. Defaults to True.
        leave_one_out (bool, optional): Whether or not to do Leave-one-out testing, if this is set to true, you must pass in an inference and score function. Defaults to False.
        plot_loss_curve (bool, optional): Whether or not to plot a loss curve for each iteration of CV, does not work with leave-one-out testing. Defaults to False.

    Raises:ape[1])
                features_i = features[i] * (1 + noise)
        ValueError: If leave_one_out is True and inference or score is None (since we need to make predictions on the test set and score them)

    Returns:
        list: A list of models from each fold
    """
    np.random.seed(42)
    models = []
    scores = []
    losses = []
    if trial is not None:
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5),
            'min_split_loss': trial.suggest_float('min_split_loss', 0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            # 'max_leaves': trial.suggest_int('max_leaves', 2, 512),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'max_depth':trial.suggest_int('max_depth', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
            'grow_policy': 'lossguide',
            'n_jobs': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': 312987524
        }
    else:
        params = None
    if preprocess is not None:
        df = preprocess(df)
    # Leave one out testing (for every data point, train on all other data points and test on that data point)
    if not leave_one_out:
        models, scores, losses = _nFoldCV(df, preprocess, model, score, n_iterations, n_folds, stratify, params=params, plot_loss_curve=plot_loss_curve)
        if score is not None:
            plt.bar(range(len(scores)), scores)
            plt.title('Validation Scores')
            plt.xlabel('Fold')
            plt.ylabel('Score')
            plt.show()
    else:
        if inference is None or score is None:
            raise ValueError('Inference and score functions must be provided for leave one out testing')
        
        loo_scores = []
        for i in range(len(df)):
            # Leave out one data point
            X_test = df.iloc[[i]]
            Y_test = pd.DataFrame(X_test.pop('Class'))
            X_df = df.drop(i)
            
            # Overwrite models scores and losses every time since we only leave out one data point
            # so the models should be about the same
            models, scores, losses = _nFoldCV(X_df, preprocess, model, score, n_iterations, n_folds, stratify, params)
            val_loss = [loss[1] for loss in losses]
            best_model_index = val_loss.index(min(val_loss))
            best_model = models[best_model_index]
            y_pred = inference(X_test, best_model)
            scores = score(Y_test, y_pred) # type: ignore
            loo_scores.append(*scores)
        
        # Plot scores
        plt.bar(range(len(df)), loo_scores)
        plt.title('Leave One Out Scores')
        plt.xlabel('Data Point')
        plt.ylabel('Score')
        plt.show()
        print('Average LOO Score:', np.mean(loo_scores))
        print('Median LOO Score:', np.median(loo_scores))
        
    val_loss = [loss[1] for loss in losses]
    train_loss = [loss[0] for loss in losses]
    
    off = 1
    # Plot scores and losses
    if score is not None:
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        sorted_scores = sorted(scores, reverse=True)
        axs[0].bar(range(len(sorted_scores)), sorted_scores)
        axs[0].set_title('Validation Scores', loc='right')
        axs[0].set_xlabel('Fold')
        axs[0].set_ylabel('Score')
    else:
        fig, axs = plt.subplots(2, 1, figsize=(10,10))
        off = 0
    sorted_val_loss = sorted(val_loss)
    sorted_train_loss = sorted(train_loss)
    axs[off].bar(range(len(sorted_val_loss)), sorted_val_loss)
    axs[off].set_title('Validation Losses',loc='right')
    axs[off].set_xlabel('Fold')
    axs[off].set_ylabel('Loss')
    axs[1+off].bar(range(len(sorted_train_loss)), sorted_train_loss)
    axs[1+off].set_title('Train Losses',loc='right')
    axs[1+off].set_xlabel('Fold')
    axs[1+off].set_ylabel('Loss')
    plt.tight_layout()
    
    # Print scores and losses
    if score is not None:
        print('Average Validation Score:', np.mean(scores))
        print('Median Validation Score:', np.median(scores))
    print('Average Validation Loss:', np.mean(val_loss))
    print('Median Validation Loss:', np.median(val_loss))
    print('Average Training Loss:', np.mean(train_loss))
    print('Median Training Loss:', np.median(train_loss))
    plt.show()
    # Return models
    if trial is not None:
        return sum(val_loss)/len(val_loss)
    return models
    
def contrastiveXGBoost(X_train, Y_train, X_val, Y_val, num_iterations, params=None):
    from DataProcessing.generatePairs import generate_pairs

    # Generate pairs
    X_train, Y_train = generate_pairs(X_train, Y_train, add_noise=False)
    X_val, Y_val = generate_pairs(X_val, Y_val, add_noise=False)
    # Shuffle X_train and Y_train together
    perm = np.random.RandomState(seed=42).permutation(len(X_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]
    
    # Train XGBoost model
    
    # xgb_params = {
    #         'learning_rate': 0.01,
    #         'max_depth': 7,
    #         'lambda': 1.3,
    #         'alpha':.6,
    #         'colsample_bytree':.4,
    #         'grow_policy': 'lossguide',
    #         'n_jobs': 24,
    #         'objective': 'binary:logistic',
    #         'eval_metric': 'logloss',
    #         'verbosity': 0,
    #         'random_state': 312987524,
    #     }
    if params is not None:
        xgb_params = params
    xgb.set_config(verbosity=0)

    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dval = xgb.DMatrix(X_val, label=Y_val)
    res = {}
    
    evallist = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(xgb_params, dtrain, num_boost_round=num_iterations, evals=evallist, early_stopping_rounds=100, evals_result=res, verbose_eval=False)

    # Make predictions on validation set
    y_pred = model.predict(dval)
    y_pred = np.round(y_pred)
    return res['train']['logloss'], res['validation']['logloss'], y_pred, model

# Read in data
data_df = pd.read_csv('./data/cleaned_train.csv')

# models: list[xgb.Booster] = train(data_df, contrastiveXGBoost, plot_loss_curve=True) # type: ignore
# for i in range(len(models)):
#     models[i].save_model(f'contrastive_{i}.json')

def train_wrapper(trial):
    return train(data_df, contrastiveXGBoost, trial=trial)

study = optuna.create_study(direction='minimize')
study.optimize(train_wrapper, n_trials=20, n_jobs=24)