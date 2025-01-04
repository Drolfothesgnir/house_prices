import numpy as np

def calculate_row_entropy(row):
    probabilities = row / row.sum()
    # Calculate entropy, ignoring zero probabilities
    entropy = -np.sum(probabilities * np.log2(probabilities + (probabilities == 0)))
    return entropy