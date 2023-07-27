import math
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

from scipy.stats import norm

from sklearn.neighbors import KernelDensity
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skewtest
from sklearn.preprocessing import KBinsDiscretizer  

# def discretize(train):

    # transformed_data = pd.DataFrame(
    #     data = PowerTransformer(method='box-cox').fit_transform(train.filter(regex='^(?!Class)')),
    #     index = df.index,
    #     columns= df.columns.drop('Class')
    # )

    # res = skewtest(transformed_data,axis=0,nan_policy='omit')

    # irreg_cols = transformed_data.columns[res.pvalue < 0.05].to_list()
    # print(f'{len(irreg_cols)} non-Gaussian features after Box-Cox:',irreg_cols)
    # feat_transformer = make_column_transformer(
    #     (KBinsDiscretizer(),irreg_cols),
    #     remainder=PowerTransformer(method='box-cox')
    # )
    # feat_transformer.fit(train)


# def kd(train):
#     bimodalNames = ['AZ','BQ','CW ','EL','GL']
#     print(f"bimodal cols: {bimodalNames}")
#     bimodal = train[bimodalNames]
#     bimodal = bimodal.fillna(bimodal.mean())
#     kmeans = KMeans(n_clusters=2)
#     for i,col in enumerate(bimodal.columns.tolist()):
#         x = bimodal[col]
#         hist = plt.hist(bimodal[col], bins=30)
#         plt.show()
#         x_d = np.linspace(-4,8,1000)
#         density = sum((abs(xi - x_d) < .5) for xi in x)
#         plt.fill_between(x_d, density, alpha=.5)
#         plt.plot(x,np.full_like(x,-.1), '|k')
#         plt.show()
#         kde = KernelDensity(kernel='gaussian').fit(bimodal[col])
#         print(kde)

# def findBimodal(train):
#     #remove Id, Class columns
#     features = train.columns.tolist()
#     features = [feature for feature in features if feature not in ['Class', 'Id']]
#     numeric = train[features].select_dtypes(include=['number'])
#     transformed = abs(numeric - numeric.mean())
#     shapiro = {feature : stats.shapiro(transformed[feature]).statistic for feature in transformed.columns}
#     figures = []
#     titles = []
#     for key in shapiro:
#         # if shapiro[key] == None:
#         #     continue
#         if shapiro[key] < .2:
#             # sns.kdeplot(data = transformed, fill=True, x=key, color='#9E3F00', legend = False)
#             print(f"col: {key}, shapiro score: {shapiro[key]}")


def twoClusters(train):
    bimodalNames = ['AZ','BQ','CW ','EL','GL']
    print(f"bimodal cols: {bimodalNames}")
    bimodal = train[bimodalNames]
    bimodal = bimodal.fillna(bimodal.mean())
    kmeans = KMeans(n_clusters=2)
    for i,col in enumerate(bimodal.columns.tolist()):
        kmeans.fit(bimodal[col])
        print(kmeans.labels_)

def findPeaks(train):
    bimodalNames = ['AZ','BQ','CW ','EL','GL']
    print(f"bimodal cols: {bimodalNames}")
    bimodal = train[bimodalNames]
    bimodal = bimodal.fillna(bimodal.mean())
    for i,col in enumerate(bimodal.columns.tolist()):
        print(bimodal[col])
        colRange = bimodal[col].max()-bimodal[col].min()
        # divide range by constant for minimum horiz distance to find peaks
        k = 2
        dist = colRange/k
        peaks = find_peaks(bimodal[col],distance=dist)
        print(f"{col}: {peaks}")

def splitBimodal(train):
    bimodalNames = ['AZ','BQ','CW ','EL','GL']
    print(f"bimodal cols: {bimodalNames}")
    bimodal = train[bimodalNames]
    bimodal = bimodal.fillna(bimodal.mean())
    for i,col in enumerate(bimodal.columns.tolist()):
        print(col)
        col_log = str(col) + "_log"
        bimodal[col_log] = bimodal[col].apply(np.log)
        print(bimodal[col_log])
        # ax = plt.subplot(5,5,2*i+1)
        # sns.histplot(data=bimodal[col_log], fill = True, x = bimodal[col_log], color = '#FEF030', legend = False)
        # ax.set_title(f"{str(col) + '_log'}", loc='right', weight = 'bold', fontsize=10)
        # ax = plt.subplot(5,5,2*i+2)  
        # sns.histplot(data=bimodal[col], fill = True, x = bimodal[col], color = '#FEF030', legend = False)
        # ax.set_title(f"{col}", loc='right', weight = 'bold', fontsize=10)

        v, b = pd.qcut(bimodal[col_log], 2, duplicates='drop', retbins=True)
        # print(f"v {v}")
        # for val in v:
        #     print(val)
        # print(f"b {b}")
        for i, row in enumerate(bimodal[col_log].items()):
            print(f"row: {row[1]}, bins: {b}")
            interval_left = pd.Interval(b[0],b[1],closed='right')
            interval_right = pd.Interval(b[1],b[2],closed='right')
            print(f"interval_left: {interval_left}")
            print(f"interval_right: {interval_right}")
            bimodal.loc[i, str(col) + '_half1'] = row[1] if row[1] in interval_left else 0
            bimodal.loc[i, str(col) + '_half2'] = row[1] if row[1] in interval_right else 0  
    figsize = (20, 30)
    fig = plt.figure(figsize=figsize)
    fig.suptitle('bimodal splits', ha='center', fontsize=21)
    for idx, col in enumerate(bimodal):
        ax = plt.subplot(5,5, idx + 1)
        sns.histplot(data=bimodal, fill = True, x = col, color = '#FEF030', legend = False)
        ax.set_title(f"{col}", loc='right', weight = 'bold', fontsize=10)
    plt.show()

    figsize = (16, 20)
    fig = plt.figure(figsize=figsize)
    for idx, col in enumerate(bimodal):
        ax = plt.subplot(11,5, idx + 1)
        sns.histplot(data=bimodal, fill = True, x = col, color = '#FEF030', legend = False)
        ax.set_title(f"{col}", loc='right', weight = 'bold', fontsize=10)
        # ax2 = plt.subplot(11,5, idx + 1)
        # sns.scatterplot(data=bimodal, y=col, color = '#FEF030', legend = False)
        # ax2.set_title(f"{col}", loc='right', weight = 'bold', fontsize=10)
    fig.suptitle('bimodal splits', ha='center', fontsize=21)
    plt.tight_layout()
    plt.show()

input_file = "./data/train.csv"
df = pd.read_csv(input_file)
# splitBimodal(df)
splitBimodal(df)