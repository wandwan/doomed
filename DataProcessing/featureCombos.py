import numpy
import pandas

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