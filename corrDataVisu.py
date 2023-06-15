import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from DataProcessing.convToNum import convert_to_numeric

# Read in input file as a dataframe
df = pd.read_csv("data/train.csv")

# Convert non-numeric columns to numeric
df = convert_to_numeric(df)

# Calculate correlation matrix
corr_matrix = df.corr().abs()

# Create heatmap of correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Find pairs of columns with correlation greater than 0.7
correlated_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j]) for i in range(corr_matrix.shape[0]) for j in range(i+1, corr_matrix.shape[0]) if corr_matrix.iloc[i,j] > 0.7]

# Calculate correlation between each correlated pair and the "Class" column
for pair in correlated_pairs:
    corr = df[pair[0]].corr(df[pair[1]])
    print(f"Correlation between {pair[0]} and {pair[1]}: {corr}")
    corr_class = df[[pair[0], pair[1], "Class"]].corr()["Class"]
    print(f"Correlation between each ({pair[0]}, {pair[1]}) and Class: \n{corr_class}")
    corr_residue = (df[pair[0]] - df[pair[1]]).corr(df["Class"])
    print(f"Correlation between residue of {pair[0]} and {pair[1]} and Class: {corr_residue}")
    
    
