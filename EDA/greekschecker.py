import pandas as pd

train_df = pd.read_csv('data/train.csv')
greeks_df = pd.read_csv('data/greeks.csv')

if not train_df['Id'].equals(greeks_df['Id']):
    print("Id columns are mismatched in order")
else:
    print("Id columns are in the same order")