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
                if feature1 != feature2:
                     combos.append(feature1 + '/' + feature2)
                     combos.append(feature1 + '*' + feature2)
                     combos.append(feature1 + '*' + feature2 + '_2')
                     combos.append(feature1 + '_2/' + feature2)
                     combos.append(feature1 + '/' + feature2 + '_2')
                     combos.append(feature1 + '_2*' + feature2 + '_2')
         combos = list(set(combos))
         return combos


# Read in train.csv
df = pd.read_csv("train.csv")

# Generate feature combinations
combos = generate_combo_features(df)

# Add columns with feature combinations
df = combo_features(df, combos)

# Calculate correlation scores between generated features and df["Class"]
from scipy.stats import pointbiserialr

corr_scores = []
for col in df.columns:
    if col != "Class":
        corr, _ = pointbiserialr(df[col], df["Class"])
        corr_scores.append((col, abs(corr)))
# corr_scores = pd.DataFrame(corr_scores, columns=["Feature", "Correlation"]).sort_values(by="Correlation", ascending=False).reset_index(drop=True)

# Plot the top 200 linear correlation scores as a line plot
corr_scores[:200].plot(kind="line")
plt.show()
