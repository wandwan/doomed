import pandas as pd
from DataProcessing.generatePairs import generate_pairs

# read in data
input = "./data/cleaned_train.csv"

# create dataframe
df = pd.read_csv(input)

