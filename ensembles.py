import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,normalize
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import xgboost
import inspect
from collections import defaultdict
from tabpfn import TabPFNClassifier
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
sample = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
greeks = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')

first_category = train.EJ.unique()[0]
train.EJ = train.EJ.eq(first_category).astype('int')
test.EJ = test.EJ.eq(first_category).astype('int')

def random_under_sampler(df):
    neg, pos = np.bincount(df['Class'])
    one_df = df.loc[df['Class'] == 1] 
    zero_df = df.loc[df['Class'] == 0]
    zero_df = zero_df.sample(n=pos)
    undersampled_df = pd.concat([zero_df, one_df])
    return undersampled_df.sample(frac = 1)
train_good = random_under_sampler(train)
predictor_columns = [n for n in train.columns if n != 'Class' and n != 'Id']
x= train[predictor_columns]
y = train['Class']
from sklearn.model_selection import KFold as KF, GridSearchCV
cv_outer = KF(n_splits = 10, shuffle=True, random_state=42)
cv_inner = KF(n_splits = 5, shuffle=True, random_state=42)
def balanced_log_loss(y_true, y_pred):
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0))
    log_loss_1 = -np.sum(y_true * np.log(p_1))
    balanced_log_loss = 2*(w_0 * log_loss_0 + w_1 * log_loss_1) / (w_0 + w_1)
    return balanced_log_loss/(N_0+N_1)
class Ensemble():
    def __init__(self):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='median')

        self.classifiers =[xgboost.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.2,subsample=0.9,colsample_bytree=0.85),
                           xgboost.XGBClassifier(),
                           TabPFNClassifier(N_ensemble_configurations=24),
                          TabPFNClassifier(N_ensemble_configurations=64)]
    
    def fit(self,X,y):
        y = y.values
        unique_classes, y = np.unique(y, return_inverse=True)
        self.classes_ = unique_classes
        first_category = X.EJ.unique()[0]
        X.EJ = X.EJ.eq(first_category).astype('int')
        X = self.imputer.fit_transform(X)
#         X = normalize(X,axis=0)
        for classifier in self.classifiers:
            if classifier==self.classifiers[2] or classifier==self.classifiers[3]:
                classifier.fit(X,y,overwrite_warning =True)
            else :
                classifier.fit(X, y)
     
    def predict_proba(self, x):
        x = self.imputer.transform(x)
#         x = normalize(x,axis=0)
        probabilities = np.stack([classifier.predict_proba(x) for classifier in self.classifiers])
        averaged_probabilities = np.mean(probabilities, axis=0)
        class_0_est_instances = averaged_probabilities[:, 0].sum()
        others_est_instances = averaged_probabilities[:, 1:].sum()
        # Weighted probabilities based on class imbalance
        new_probabilities = averaged_probabilities * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) for i in range(averaged_probabilities.shape[1])]])
        return new_probabilities / np.sum(new_probabilities, axis=1, keepdims=1) 

from tqdm.notebook import tqdm

def training(model, x,y,y_meta):
    outer_results = list()
    best_loss = np.inf
    split = 0
    splits = 5
    for train_idx,val_idx in tqdm(cv_inner.split(x), total = splits):
        split+=1
        x_train, x_val = x.iloc[train_idx],x.iloc[val_idx]
        y_train, y_val = y_meta.iloc[train_idx], y.iloc[val_idx]
                
        model.fit(x_train, y_train)
        y_pred = model.predict_proba(x_val)
        probabilities = np.concatenate((y_pred[:,:1], np.sum(y_pred[:,1:], 1, keepdims=True)), axis=1)
        p0 = probabilities[:,:1]
        p0[p0 > 0.86] = 1
        p0[p0 < 0.14] = 0
        y_p = np.empty((y_pred.shape[0],))
        for i in range(y_pred.shape[0]):
            if p0[i]>=0.5:
                y_p[i]= False
            else :
                y_p[i]=True
        y_p = y_p.astype(int)
        loss = balanced_log_loss(y_val,y_p)

        if loss<best_loss:
            best_model = model
            best_loss = loss
            print('best_model_saved')
        outer_results.append(loss)
        print('>val_loss=%.5f, split = %.1f' % (loss,split))
    print('LOSS: %.5f' % (np.mean(outer_results)))
    return best_model
    
from datetime import datetime
times = greeks.Epsilon.copy()
times[greeks.Epsilon != 'Unknown'] = greeks.Epsilon[greeks.Epsilon != 'Unknown'].map(lambda x: datetime.strptime(x,'%m/%d/%Y').toordinal())
times[greeks.Epsilon == 'Unknown'] = np.nan

train_pred_and_time = pd.concat((train, times), axis=1)
test_predictors = test[predictor_columns]
first_category = test_predictors.EJ.unique()[0]
test_predictors.EJ = test_predictors.EJ.eq(first_category).astype('int')
test_pred_and_time = np.concatenate((test_predictors, np.zeros((len(test_predictors), 1)) + train_pred_and_time.Epsilon.max() + 1), axis=1)

# %%
ros = RandomOverSampler(random_state=42)

train_ros, y_ros = ros.fit_resample(train_pred_and_time, greeks.Alpha)
print('Original dataset shape')
print(greeks.Alpha.value_counts())
print('Resample dataset shape')
print( y_ros.value_counts())

# %%
x_ros = train_ros.drop(['Class', 'Id'],axis=1)
y_ = train_ros.Class

# %%
yt = Ensemble()

# %%
m = training(yt,x_ros,y_,y_ros)

# %%
y_.value_counts()/y_.shape[0]

# %%
y_pred = m.predict_proba(test_pred_and_time)
probabilities = np.concatenate((y_pred[:,:1], np.sum(y_pred[:,1:], 1, keepdims=True)), axis=1)
p0 = probabilities[:,:1]
p0[p0 > 0.59] = 1 
p0[p0 < 0.28] = 0

# %%
submission = pd.DataFrame(test["Id"], columns=["Id"])
submission["class_0"] = p0
submission["class_1"] = 1 - p0
submission.to_csv('submission.csv', index=False)

# %%
submission_df = pd.read_csv('submission.csv')
submission_df



