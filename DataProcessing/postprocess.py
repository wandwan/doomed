from scipy.optimize import minimize
import numpy as np


def balanced_logloss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    n0 = (y_true==0).sum()
    n1 = (y_true==1).sum()
    logloss = ( -1/n0*np.log(1-y_pred[y_true==0]).sum()  - 1/n1*np.log(y_pred[y_true==1]).sum() ) / 2
    return logloss


def adj_pred_logloss(k, pred):
    t = 1.0 * ( pred > np.random.uniform(0,1,size=len(pred)))
    odds = pred / ( 1 - pred )
    adj_odds = k[0] * odds
    adj_pred = adj_odds / ( 1 + adj_odds )
    return balanced_logloss(t, adj_pred)

def post(pred):
    foo = minimize(adj_pred_logloss, x0=[1], args=(pred))
    return foo.x
