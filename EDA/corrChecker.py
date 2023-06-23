import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./data/cleaned_train.csv")
colors = df["Class"]

def assign_colors(df):
    colors = []
    for c in df["Class"]:
        if c == 1:
            colors.append("red")
        else:
            colors.append("blue")
    return colors

DU_l = np.log1p(df["DU"])
mat = pd.concat([DU_l/df["FI"], DU_l/df["EP"], DU_l*df["AR"]], axis=1)
print(mat)
corr_matrix = mat.corr()
print(corr_matrix)

