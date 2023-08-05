import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from random import sample

# Define a function to generate pairs for a given row
def generate_pairs_for_row(i,fraction, frac, features, n, add_noise, labels):
    row_pairs = np.empty((frac, features.shape[1] * 3))
    row_labels = np.zeros((frac, 2), dtype=np.int32)
    kept = sample(range(0,n),(int) (n * fraction))
    for idx, j in enumerate(kept):
        if add_noise:
            # Add noise to the features
            noise = np.random.normal(loc=0.0, scale=0.15, size=features.shape[1])
            features_i = features[i] * (1 + noise)
            features_j = features[j] * (1 + noise)
            row_pairs[idx] = np.concatenate([features_i - features_j, features_i / np.clip(features_j, 1e-6, None), features_j])
        else:
            # Perform element-wise addition, subtraction, and multiplication of the features of two rows
            row_pairs[idx] = np.concatenate([features[i] - features[j], features[i] / np.clip(features[j], 1e-6, None), features[j]], dtype=float)
            # row_pairs[idx] = np.concatenate([features[i] - features[j], features[j]], dtype=float)

        # Add the XOR of the two classes to the labels array
        row_labels[idx][labels[i] ^ labels[j]] = 1
    return row_pairs, row_labels, i

def generate_pairs(features, labels, fraction=1, add_noise=False):
    # Get the number of rows in the features array
    n = features.shape[0]
    frac = ((int) (n * fraction))
    # Create an empty numpy array to store the pairs
    pairs = np.empty((n * frac , 3 * features.shape[1]))
    
    # Create an empty numpy array to store the labels
    labels_array = np.zeros((n * frac, 2), dtype=np.int32)
    
    # Generate pairs for each row in parallel
    results = Parallel(n_jobs=-1)(delayed(generate_pairs_for_row)(i, fraction, frac, features, n, add_noise, labels) for i in range(n)) #type: ignore
    
    # Combine the results into the final pairs and labels arrays
    idx = 0
    for row_pairs, row_labels, _ in results: #type: ignore
        pairs[idx:idx+frac] = row_pairs
        labels_array[idx:idx+frac] = row_labels
        idx += frac
    
    return pairs, labels_array

def generate_pairs_for_val_row(i, features, labels, val_features, val_labels):
    # Get the number of rows in the features array
    n = features.shape[0]
    # Create an empty numpy array to store the pairs
    pairs = np.empty((n, 3 * features.shape[1]))
    
    # Create an empty numpy array to store the labels
    labels_array = np.zeros((n, 2), dtype=np.int32)
    
    # Generate pairs for each row in parallel
    for j in range(n):
        # Perform element-wise addition, subtraction, and multiplication of the features of two rows
        pairs[j] = np.concatenate([features[j] - val_features[i], features[j] / np.clip(val_features[i], 1e-6, None), val_features[i]], dtype=float)
        # Add the XOR of the two classes to the labels array
        labels_array[j][labels[j] ^ val_labels[i]] = 1

def generate_pairs_for_validation(train_features, val_features, train_labels, val_labels):
    # Get the number of rows in the features array
    n = train_features.shape[0]
    frac = n
    # Create an empty numpy array to store the pairs
    pairs = np.empty((n * frac , 3 * train_features.shape[1]))
    
    # Create an empty numpy array to store the labels
    labels_array = np.zeros((n * frac, 2), dtype=np.int32)
    
    # Generate pairs for each row in parallel
    results = Parallel(n_jobs=-1)(delayed(generate_pairs_for_val_row)(i, train_features, train_labels, val_features, val_labels) for i in range(n)) #type: ignore
    results.sort(key=lambda x: x[2])
    # Combine the results into the final pairs and labels arrays
    idx = 0
    for row_pairs, row_labels, _ in results:
        pairs[idx:idx+frac] = row_pairs
        labels_array[idx:idx+frac] = row_labels
        idx += frac