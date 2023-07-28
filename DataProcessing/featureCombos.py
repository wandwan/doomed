from scipy.stats import pointbiserialr # type: ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from DataProcessing.norm import logAndScaleColumns
import pickle
import re

def _transformCol(df, col):
    if "2" in col:
        col = col[:2:]
        return df[col] ** 2
    elif "l" in col:
        col = col[:2:]
        return np.log1p(df[col])
    return df[col]
def combo_features(df, selected):
    # Add columns with feature combinations
    for feature in selected:
        col1, col2 = "", ""
        div = True
        if '/' in feature:
            col1, col2 = feature.split('/')
        elif '*' in feature:
            col1, col2 = feature.split('*')
            div = False
        col1, col2 = _transformCol(df, col1), _transformCol(df, col2)
        if div:
            df[feature] = col1 / col2
        else:
            df[feature] = col1 * col2
    return df.copy()

def generate_combo_features(df: pd.DataFrame, verbose=False) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, float], dict[str, pd.DataFrame]]:
        # Create a list of all possible feature combinations
        features = df.columns
        combos = []
        seen = set()
        for feature1 in features:
            for feature2 in features:
                if feature1 != feature2 and feature1 not in ["Id", "Class"] and feature2 not in ["Id", "Class"]:
                    combos.append((feature1 + '/' + feature2, (df[feature1] / df[feature2]).replace([np.inf, -np.inf], 10000)))
                    combos.append((feature1 + '*' + feature2, df[feature1] * df[feature2]))
                    combos.append((feature1 + '*' + feature2 + '_2', df[feature1] * df[feature2] ** 2))
                    combos.append((feature1 + '_2/' + feature2, (df[feature1] ** 2 / df[feature2]).replace([np.inf, -np.inf], 10000)))
                    combos.append((feature1 + '/' + feature2 + '_2', (df[feature1] / df[feature2] ** 2).replace([np.inf, -np.inf], 1000)))
                    if feature2+feature1 not in seen:
                        combos.append((feature1 + '_2*' + feature2 + '_2', df[feature1] ** 2 * df[feature2] ** 2))           
                        seen.add(feature1+feature2)
                    if feature1 != "" and feature2 != "":
                        combos.append((feature1 + '_l/' + feature2, (np.log1p(df[feature1]) / df[feature2]).replace([np.inf, -np.inf], 10000)))
                        combos.append((feature1 + '_l*' + feature2, np.log1p(df[feature1]) * df[feature2]))
                        combos.append((feature1 + '/' + feature2 + '_l', (df[feature1] / np.log1p(df[feature2])).replace([np.inf, -np.inf], 10000)))
                        if feature2+feature1 not in seen:
                            combos.append((feature1 + '_l*' + feature2 + '_l', np.log1p(df[feature1]) * np.log1p(df[feature2])))
                            seen.add(feature1+feature2)
        # Calculate correlation scores between generated features and df["Class"]

        col_corrs = {}
        for col in df.columns:
            if col not in ["Id", "Class"]:
                corr, _ = pointbiserialr(df["Class"], df[col])
                col_corrs[col] = corr
        mapper = {}
        # calculate excess correlation
        corr_scores = []
        for col in combos:
            mapper[col[0]] = col[1]
            corr, _ = pointbiserialr(df["Class"], col[1])
            name1, name2 = re.split("/|\*", col[0])
            name1, name2 = name1[:2:], name2[:2:]
            corr_scores.append((col[0], abs(corr ** 2) - ((abs(col_corrs[name1]) ** 2 + abs(col_corrs[name2]) ** 2))))
        corr_scores = pd.DataFrame(corr_scores, columns=["Feature", "ExcessCorr"]).sort_values(by="ExcessCorr", ascending=False).reset_index(drop=True)
        
        # corr_scores contains the excess correlation of each feature combination
        # col_corrs contains the point_biserial correlation of each feature with df["Class"]
        # mapper maps the feature name to the actual generated feature vals (string to df[col])
        if verbose:
            return corr_scores, col_corrs, mapper
        return corr_scores

def calculateNewFeatures(df, numNewFeaturesKept=100):
    # Generate new features
    corr_scores: pd.DataFrame
    mapper: dict[str, pd.DataFrame]
    corr_scores, _, mapper = generate_combo_features(df, True) # type: ignore
    print("Generated new features")
    print(corr_scores["ExcessCorr"].head(60))

    # Keep only the top 1000 rows
    kept = []
    for idx, row in corr_scores.iterrows():
        if int(idx) > numNewFeaturesKept: # type: ignore
            break
        kept.append(row["Feature"])
        for i in range(len(kept) - 1):
            if pd.concat([mapper[kept[-1]], mapper[kept[i]]], axis=0).corr() > .75:
                kept.pop()
                break
    
    # Convert mapper to dataframe
    new_features = pd.DataFrame().from_dict({key: mapper[key] for key in kept})
    return new_features
    

# The following functions are so shit, just remove them eventually please
def visualize():
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
    corr_scores, col_corrs, mapper = generate_combo_features(df, True)

    print("finished generating combos")

    import seaborn as sns
    figsize = (6*6, 20)
    fig = plt.figure(figsize=figsize)
    for idx, row in corr_scores.iterrows(): # type: ignore
        if idx > 9: # type: ignore
            break
        copy = {"Feature": mapper[row["Feature"]], "Class": df["Class"]} # type: ignore
        copy = pd.DataFrame(copy)
        
        ax = plt.subplot(5,6, 3 * idx + 1) # type: ignore
        sns.kdeplot(copy[copy["Class"] == 1]["Feature"], fill=True, legend=False)
        sns.kdeplot(copy[copy["Class"] == 0]["Feature"], fill=True, legend=False)
        ax.set_title(f'{row["Feature"]}', loc='right')

        name1, name2 = re.split("/|\*", row["Feature"])
        name1, name2 = name1[:2:], name2[:2:]
        
        ax = plt.subplot(5,6, 3 * idx + 2) # type: ignore
        sns.kdeplot(np.log1p(df[df["Class"] == 1][name1]), fill=True, legend=False)
        sns.kdeplot(df[df["Class"] == 0][name1], fill=True, legend=False)
        ax.set_title(f'{name1}', loc='right')
        
        ax = plt.subplot(5,6, 3 * idx + 3) # type: ignore
        sns.kdeplot(df[df["Class"] == 1][name2], fill=True, legend=False)
        sns.kdeplot(df[df["Class"] == 0][name2], fill=True, legend=False)
        ax.set_title(f'{name2}', loc='right')

    fig.suptitle(f'ExcessCorr vs Features\n\n\n', ha='center',  fontweight='bold', fontsize=21)
    fig.legend([1, 0], loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=21, ncol=3)
    plt.tight_layout()
    plt.show()


    print(corr_scores)
    corr_scores.to_csv("out.csv",index=True) # type: ignore

    import csv
    with open('corrs.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, col_corrs.keys()) # type: ignore
        w.writeheader()
        w.writerow(col_corrs) # type: ignore
        
    for idx, row in corr_scores.iterrows(): # type: ignore
        name1, name2 = re.split("/|\*", row["Feature"])
        name1, name2 = name1[:2:], name2[:2:]
        print("Feature: {}, ExcessCorr: {}, individual: {}, {}".format(row["Feature"], row["ExcessCorr"], col_corrs[name1], col_corrs[name2])) # type: ignore
        if(idx > 200): # type: ignore
            break
        

    # Plot the top 200 linear correlation scores as a line plot
    corr_scores[:200].plot(kind="line") # type: ignore
    plt.show()

# calculate the correlation between any 2 combination of created features
# This function is a piece of shit, ignore.
def created_features_corr():
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
    corr_scores, col_corrs, mapper = generate_combo_features(df, True)

    print("finished generating combos")
    
    #keep only the top 200 rows
    kept = []
    for idx, row in corr_scores.iterrows(): # type: ignore
        kept.append(row["Feature"])
        if idx > 200:
            break

    #remove all columns that are not in the top 200
    for key in list(mapper.keys()):
        if key not in kept:
            del mapper[key]
    
    # Convert mapper dictionary to dataframe
    mapper_df = pd.DataFrame.from_dict(mapper)
    
    # Calculate correlation between all columns in mapper_df
    corr_matrix = mapper_df.corr()
    
    # Show correlation as plot
    plt.matshow(corr_matrix)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.colorbar()
    plt.show()

