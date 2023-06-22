import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from random import sample

def generate_pairs(features, labels, fraction=1, add_noise=True):
    # Get the number of rows in the features array
    n = features.shape[0]
    frac = ((int) (n * fraction))
    # Create an empty numpy array to store the pairs
    pairs = np.empty((n * frac , 3 * features.shape[1]))
    
    # Create an empty numpy array to store the labels
    labels_array = np.zeros((n * frac, 2), dtype=np.int32)
    
    # Define a function to generate pairs for a given row
    def generate_pairs_for_row(i,fraction):
        row_pairs = np.empty((frac, features.shape[1] * 3))
        row_labels = np.zeros((frac, 2), dtype=np.int32)
        kept = sample(range(0,n),(int) (n * fraction))
        for idx, j in enumerate(kept):
            if add_noise:
                # Add noise to the features
                noise = np.random.normal(loc=0.0, scale=0.15, size=features.shape[1])
                features_i = features[i] * (1 + noise)
                features_j = features[j] * (1 + noise)
                row_pairs[idx] = np.concatenate([features_i - features_j, features_i / (features_j + 1), features_j])
            else:
                # Perform element-wise addition, subtraction, and multiplication of the features of two rows
                row_pairs[idx] = np.concatenate([features[i] - features[j], features[i] / (features[j] + 1), features[j]])
            # Add the XOR of the two classes to the labels array
            row_labels[idx][labels[i] ^ labels[j]] = 1
        return row_pairs, row_labels, i
    
    # Generate pairs for each row in parallel
    results = Parallel(n_jobs=1)(delayed(generate_pairs_for_row)(i, fraction) for i in range(n)) #type: ignore
    results.sort(key=lambda val: val[2]) #type: ignore
    
    # Combine the results into the final pairs and labels arrays
    idx = 0
    for row_pairs, row_labels, _ in results: #type: ignore
        pairs[idx:idx+frac] = row_pairs
        labels_array[idx:idx+frac] = row_labels
        idx += frac
    
    return pairs, labels_array
