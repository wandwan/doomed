import logging

import numpy as np

from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer, PowerTransformer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss
import pandas as pd

from splitData import split_data

logger = logging.Logger('SVM', logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:\n%(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

def shift_to_one(d: pd.DataFrame) -> pd.DataFrame:
  """
  Shifts the range of the df so the min is 1 for each col
  """

  for col in d.columns:
    min = np.min(d[col])
    d[col] -= min - 1
  return d

def balanced_log_loss(y_true, y_pred) -> float:
  y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
  n0 = (y_true==0).sum()
  n1 = (y_true==1).sum()
  logloss = (-1/n0*np.log(1-y_pred[y_true==0]).sum() - 1/n1*np.log(y_pred[y_true==1]).sum()) / 2
  return logloss

def build_pipeline():
    clf = make_pipeline(SimpleImputer(), PowerTransformer(method='box-cox'),
                         SVR(kernel='rbf', max_iter=10_000))
    
    logger.info(clf)
    return clf

df = pd.read_csv('data/train.csv', index_col='Id')
df.replace({'A': 1, 'B': 2}, inplace=True)

def train_SVR(pipe: StackingClassifier):
    # losses holds tuples of (train_loss, val_loss)
    losses = []
    # models holds strings of model paths or model objects
    models = []
    for X_train, X_val, Y_train, Y_val in split_data(df, 10, stratify=True):
        
        fitted = pipe.fit(X_train, Y_train)
        
        score = fitted.score(X_val, Y_val)
        logger.debug(score)

        train_pred = fitted.predict(X_train) - 1
        val_pred = fitted.predict(X_val) - 1
        X_train, X_val, Y_train, Y_val = X_train - 1, X_val - 1, Y_train - 1, Y_val - 1
        logger.debug(val_pred)
        logger.debug(Y_val)
        train_loss = balanced_log_loss(Y_train, train_pred)
        val_loss = balanced_log_loss(Y_val, val_pred)
        losses.append((train_loss, val_loss))

        models.append(fitted)
    
    return losses, models

df = shift_to_one(df)
losses, models = train_SVR(build_pipeline())

vals = [x for _, x in losses]

logger.info(losses)
logger.info(np.average(vals))
