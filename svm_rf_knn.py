import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('data/train.csv', index_col='Id')
train_df.replace({'A': 1, 'B': 2}, inplace=True)

imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(train_df))

y_train = train_df['Class']

simple_imputer = SimpleImputer()
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(simple_imputer.fit_transform(train_df))

def build_svm():
  parameters = {'C': np.logspace(-3, 3, 5),'degree':(1,2,3,4,5,6,7,8)}
  svm = SVC(kernel='poly',probability=True,)
  class_weights = {0: 82.495948, 1: 17.504052} 
  svm.set_params(class_weight=class_weights)
  svm_cv = GridSearchCV(svm, parameters, cv= 15)
  svm_cv.fit(X_train_scale, y_train)
  return svm_cv

def build_knn():
  parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}
  KNN = KNeighborsClassifier(weights='distance')
  knn_cv = GridSearchCV(KNN, parameters, cv= 10)
  knn_cv.fit(X_imputed, y_train)
  return knn_cv

def build_rf():
  # Number of trees in random forest
  n_estimators = [x for x in range(200,2000,200)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt']
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  max_depth.append(None)
  # Minimum number of samples required to split a node
  min_samples_split = [2, 5, 10]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 2, 4]
  # Method of selecting samples for training each tree
  bootstrap = [True, False]
  # Create the random grid
  random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
  rf = RandomForestClassifier()
  class_weights = {0: 82.495948, 1: 17.504052} 
  rf.set_params(class_weight=class_weights)
  # Random search of parameters, using 3 fold cross validation, 
  # search across 100 different combinations, and use all available cores
  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
  # Fit the random search model
  rf_random.fit(X_train_scale, y_train)
  return rf_random

def build_ensemble():
  knn_cv = build_knn()
  svm_cv = build_svm()
  rf_random = build_rf()
  ensemble = VotingClassifier(estimators=[('Knn', knn_cv), ('svm', svm_cv),('rf',rf_random)],n_jobs=-1,voting='soft',weights=[2,2,1])
  ensemble.fit(X_train_scale, y_train)
  return ensemble

ensemble = build_ensemble()

test_df=pd.read_csv('data/train.csv')
ID = test_df['Id']
test_df.set_index('Id', inplace=True)
test_df.loc[(test_df.EJ == 'A'), 'EJ'] = 0
test_df.loc[(test_df.EJ == 'B'), 'EJ'] = 1
colum = test_df.columns
test_df_scaler = scaler.fit_transform(test_df)
if test_df.isnull().sum().any() > 0:
    test_df_scaler = pd.DataFrame(imputer.fit_transform(test_df_scaler))
test_df = pd.DataFrame(test_df_scaler)
test_df.columns = colum
# test_df.drop(['BZ', 'EH', 'CL'], axis=1, inplace=True)
y_pred = ensemble.predict_proba(test_df)
submission_pred = pd.DataFrame(y_pred, columns=['class_0', 'class_1'])
submission_pred['Id']=ID
submission_pred=submission_pred[['Id','class_0', 'class_1']]
submission_pred.to_csv('submission.csv',index=False)