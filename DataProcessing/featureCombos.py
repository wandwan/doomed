import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def combo_features(df, selected):
       # Add columns with feature combinations
       for feature in selected:
           if '/' in feature:
               col1, col2 = feature.split('/')
               df[feature] = df[col1] / df[col2]
           elif '*' in feature:
               col1, col2 = feature.split('*')
               if '2' in col1:
                   col1, col2 = col2, col1
               if '2' in col2:
                   col2 = col2.replace('_2', '')
                   df[feature] = df[col1] ** 2 * df[col2]
               else:
                   df[feature] = df[col1] * df[col2]
       return df.copy()

def generate_combo_features(df):
         # Create a list of all possible feature combinations
         features = df.columns
         combos = []
         for feature1 in features:
              for feature2 in features:
                if feature1 != feature2 and feature1 not in ["Id", "Class"] and feature2 not in ["Id", "Class"]:
                     combos.append((feature1 + '/' + feature2, (df[feature1] / df[feature2]).replace([np.inf, -np.inf], 0)))
                     combos.append((feature1 + '*' + feature2, df[feature1] * df[feature2]))
                     combos.append((feature1 + '*' + feature2 + '_2', df[feature1] * df[feature2] ** 2))
                     combos.append((feature1 + '_2/' + feature2, (df[feature1] ** 2 / df[feature2]).replace([np.inf, -np.inf], 0)))
                     combos.append((feature1 + '/' + feature2 + '_2', (df[feature1] / df[feature2] ** 2).replace([np.inf, -np.inf], 0)))
                     combos.append((feature1 + '_2*' + feature2 + '_2', df[feature1] ** 2 * df[feature2] ** 2))
         return combos


# Read in train.csv
df = pd.read_csv("data/train.csv")

# Remove whitespace from column names
df.columns = df.columns.str.replace(' ', '')

from convToNum import convert_to_numeric
# Convert all columns to numbers
df = convert_to_numeric(df)

# Fill nan's
from fillNan import fill_na_with_kmeans
df = fill_na_with_kmeans(df)

# Generate feature combinations
combos = generate_combo_features(df)


print("finished generating combos")
print(len(combos))

# Calculate correlation scores between generated features and df["Class"]
from scipy.stats import pointbiserialr



col_corrs = {}
for col in df.columns:
    if col not in ["Id", "Class"]:
        corr, _ = pointbiserialr(df[col], df["Class"])
        col_corrs[col] = corr
import re
corr_scores = []
mapper = {}
for col in combos:
    corr, _ = pointbiserialr(col[1], df["Class"])
    name1, name2 = re.split("/|\*", col[0])
    name1, name2 = name1[:2:], name2[:2:]
    mapper[col[0]] = col[1]
    corr_scores.append((col[0], abs(corr) ** 2 - (max(abs(col_corrs[name1]), abs(col_corrs[name2])))**2))
corr_scores = pd.DataFrame(corr_scores, columns=["Feature", "Correlation"]).sort_values(by="Correlation", ascending=False).reset_index(drop=True)

import seaborn as sns
figsize = (6*6, 20)
fig = plt.figure(figsize=figsize)
for idx, row in corr_scores.iterrows():
    if idx > 9:
        break
    ax = plt.subplot(5,6, 3 * idx + 1)
    copy = {"Feature": mapper[row["Feature"]], "Class": df["Class"]}
    copy = pd.DataFrame(copy)
    
    
    sns.kdeplot(copy[copy["Class"] == 1]["Feature"])
    sns.kdeplot(copy[copy["Class"] == 0]["Feature"])
    ax.set_title(f'{row["Feature"]}')

    name1, name2 = re.split("/|\*", row["Feature"])
    name1, name2 = name1[:2:], name2[:2:]
    
    ax = plt.subplot(5,6, 3 * idx + 2)
    sns.kdeplot(df[df["Class"] == 1][name1])
    sns.kdeplot(df[df["Class"] == 0][name1])
    ax.set_title(f'{name1}')
    ax = plt.subplot(5,6, 3 * idx + 3)
    sns.kdeplot(df[df["Class"] == 1][name2])
    sns.kdeplot(df[df["Class"] == 0][name2])
    ax.set_title(f'{name1}')
plt.tight_layout()
plt.show()


print(corr_scores)
corr_scores.to_csv("out.csv",index=True)

import csv
with open('corrs.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, col_corrs.keys())
    w.writeheader()
    w.writerow(col_corrs)
    
for idx, row in corr_scores.iterrows():
    name1, name2 = re.split("/|\*", row["Feature"])
    name1, name2 = name1[:2:], name2[:2:]
    print("Feature: {}, Correlation: {}, individual: {}, {}".format(row["Feature"], row["Correlation"], col_corrs[name1], col_corrs[name2]))
    if(idx > 200):
        break
    

# Plot the top 200 linear correlation scores as a line plot
corr_scores[:200].plot(kind="line")
plt.show()
