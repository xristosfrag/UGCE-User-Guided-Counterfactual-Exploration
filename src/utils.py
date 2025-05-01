from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist
from sklearn.pipeline import Pipeline
from prettytable import PrettyTable
from scipy.spatial import distance
import pandas as pd
import numpy as np

def fast_max_l2_l1_distance(data, sample_size=40000, seed=42):
    if len(data) > sample_size:
        np.random.seed(seed)
        sample_indices = np.random.choice(len(data), sample_size, replace=False)
        sample = data[sample_indices]
    else:
        sample = data
    return np.max(pdist(sample, metric='euclidean')), np.max(pdist(sample, metric='cityblock'))


def compute_distances_in_blocks(data, block_size=1000, representation=16):
    """Computes distances in smaller blocks and rounds results to save memory.
    
    Understanding Memory Usage:
    Supposing a dataset of 40,000 samples.
    The pairwise distance matrix is 40,000 x 40,000.
    Since storing distances in a float64 (8 bytes per value), the total memory required is:
    40,000 x 40,000 x 8 bytes = 12.8 GB (approximately).
    
    Expected Memory Savings
        Data Type	Memory per Value	Full Matrix Size
        float64	    8 bytes	            12.8 GB
        float32	    4 bytes	            6.4 GB
        float16	    2 bytes	            3.2 GB
    """
    n = len(data)
    if representation == 16:
        decimals = np.float16
    elif representation == 32:
        decimals = np.float32
    elif representation == 64:
        decimals = np.float64
    
    results = np.zeros((n, n), dtype=decimals)
    for i in range(0, n, block_size):
        for j in range(i, n, block_size):
            i_end = min(i + block_size, n)
            j_end = min(j + block_size, n)
            results[i:i_end, j:j_end] = distance.cdist(data[i:i_end], data[j:j_end], 'euclidean').astype(decimals)
            if i != j:
                results[j:j_end, i:i_end] = results[i:i_end, j:j_end].T  # Symmetric assignment
    return results

def transform_individual(individual, scaler):
    return scaler.transform(individual.reshape(1, -1))

# Inverse transform a scaled individual back to the original space
def inverse_transform_individual(scaled_individual, scaler, feature_columns):
    if len(scaled_individual) != len(feature_columns):
        raise ValueError(f"Expected feature length: {len(feature_columns)}, got: {len(scaled_individual)}. Please check your feature selection pipeline.")
    original_individual = scaler.inverse_transform(scaled_individual.reshape(1, -1))
    return original_individual[0], pd.DataFrame(original_individual, columns=feature_columns).iloc[0].to_dict()

# Scale user-provided constraints from original space to scaled space
def scale_constraints(original_constraints, scaler, feature_columns):
    # Create a dummy dataframe with the constraints
    constraint_df = pd.DataFrame([original_constraints], columns=feature_columns)
    # Scale the constraints using the same scaler
    scaled_constraints = scaler.transform(constraint_df)
    return {i: (scaled_constraints[0][i], scaled_constraints[0][i]) for i in range(len(feature_columns))}

# Train a binary classification model on the COMPAS dataset
def train_test_split_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

# User-defined model function to wrap the logistic regression model
def f_model(instance, model):
    return model.predict(instance.reshape(1, -1))[0]

def display_cfe_comparison(original, cfe):
    table = PrettyTable()
    table.field_names = ["Feature", "Original Value", "Proposed CFE", "Change"]

    for feature, original_value in original.items():
        proposed_value = cfe[feature]
        # Calculate the difference for numerical changes
        if isinstance(original_value, (int, float)) and isinstance(proposed_value, (int, float)):
            change = f"{proposed_value - original_value:+}" if original_value != proposed_value else ""
        else:
            # For non-numeric values (like categorical one-hot encoded), use "->" notation if there's a change
            change = "->" if original_value != proposed_value else ""

        table.add_row([feature, original_value, proposed_value, change])

    table.align = "l"
    print("\nComparison of Initial Instance and Proposed CFE:")
    print(table)