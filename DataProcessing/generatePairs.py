import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def generate_pairs(features, labels, percent=1.0):
    # Get the number of rows in the features array
    n = features.shape[0]
    
    # Create an empty numpy array to store the pairs
    pairs = np.empty((n * n, 2 * features.shape[1]))
    
    # Create an empty numpy array to store the labels
    labels_array = np.zeros((n * n, 2), dtype=np.int32)
    
    # Define a function to generate pairs for a given row
    def generate_pairs_for_row(i):
        row_pairs = np.empty((n, 2 * features.shape[1]))
        row_labels = np.zeros((n, 2), dtype=np.int32)
        
        for j in range(n):
            # Add the pair of rows to the numpy array
            row_pairs[j] = np.concatenate([features[i], features[j]])
            
            # Add the XOR of the two classes to the labels array
            row_labels[j][labels[i] ^ labels[j]] = 1
        
        return row_pairs, row_labels
    
    # Generate pairs for each row in parallel
    results = Parallel(n_jobs=-1)(delayed(generate_pairs_for_row)(i) for i in range(n))
    
    # Combine the results into the final pairs and labels arrays
    idx = 0
    for row_pairs, row_labels in results:
        pairs[idx:idx+n] = row_pairs
        labels_array[idx:idx+n] = row_labels
        idx += n
    
    return pairs, labels_array
