import matplotlib.pyplot as plt
import numpy as np
from splitData import split_data, split_numpy_data
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import re

# Read in data
data_df = pd.read_csv('./data/cleaned_train.csv')

# Use only 100 rows for testing
data_df = data_df[:300]

# Split data into train and validation sets
X_train_df, X_val_df, Y_train_df, Y_val_df = split_data(data_df)

# Convert dataframes to numpy arrays
X_train = X_train_df.to_numpy()
X_val = X_val_df.to_numpy()
Y_train = np.round(Y_train_df.to_numpy()).astype(int)
Y_val = np.round(Y_val_df.to_numpy()).astype(int)

from DataProcessing.generatePairs import generate_pairs

# Generate pairs
X_train, Y_train = generate_pairs(X_train, Y_train)
X_val, Y_val = generate_pairs(X_val, Y_val)

# Train XGBoost model
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
    'silent': 1.0,
    'n_estimators': 100
}
num_iterations = 100

dtrain = xgb.DMatrix(X_train, label=Y_train)
dval = xgb.DMatrix(X_val, label=Y_val)
res = {}

evallist = [(dtrain, 'train'), (dval, 'validation')]
model = xgb.train(params, dtrain, num_boost_round=num_iterations, evals=evallist, early_stopping_rounds=10, evals_result=res)

# Make predictions on validation set
y_pred = model.predict(dval)
y_pred = np.round(y_pred)

# Calculate accuracy
accuracy = accuracy_score(Y_val, y_pred)
print('Accuracy:', accuracy)

# Graph train and validation loss
train_loss = res['train']['logloss']
val_loss = res['validation']['logloss']
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()