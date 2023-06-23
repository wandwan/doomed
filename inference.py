import json
import xgboost as xgb
import pandas as pd
import numpy as np

# Load the XGBoost model from the JSON file
with open('model.json', 'r') as f:
    model_json = json.load(f)
model = xgb.Booster(model_file=None, params=model_json['params'])

model.load_model('model.json')
