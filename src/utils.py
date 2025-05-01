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

# ========================================================================================================================
# ========================================================================================================================
# ========================================================================================================================
def normalize_data(data, feature_names, full_data):
    """Normalizes features based on min-max values from a reference dataset.
    
    Parameters:
    - data: The dataset to be normalized (can be list, NumPy array, or DataFrame).
    - feature_names: The names of the features to be normalized.
    - full_data: The reference dataset used to compute min-max values.
    
    Returns:
    - The normalized dataset as a pandas DataFrame (if input was DataFrame),
      or a NumPy array (if input was list or NumPy array).
    """
    if isinstance(data, list):
        data = np.array(data, dtype=float)

    min_values = full_data[feature_names].min()
    max_values = full_data[feature_names].max()

    if isinstance(data, pd.DataFrame):
        normalized_data = data.copy()
        for feature in feature_names:
            if min_values[feature] == max_values[feature]:
                normalized_data[feature] = 0
            else:
                normalized_data[feature] = (data[feature] - min_values[feature]) / (max_values[feature] - min_values[feature])
        return normalized_data
    elif isinstance(data, np.ndarray):
        normalized_data = data.astype(float)
        
        for i, feature in enumerate(feature_names):
            min_value = min_values[feature]
            max_value = max_values[feature]
            if min_value == max_value:
                normalized_data[:, i] = 0
            else:
                normalized_data[:, i] = (data[:, i] - min_value) / (max_value - min_value)
        return normalized_data
    else:
        raise ValueError("Unsupported input type. Expected pandas DataFrame, NumPy array, or list.")

def initial_label_encode_data(df, feature_names, categorical):
    """
    Label encodes categorical features in a DataFrame.

    Parameters:
    - df: The input DataFrame.
    - feature_names: List of feature names of the dataframe.
    - categorical: List of categorical feature names to be encoded.

    Returns:
    - A new DataFrame with label-encoded categorical features.
    - A dictionary of LabelEncoders used for each feature.
    """
    encoded_df = df.copy()
    label_encoders = {}

    for feature in feature_names:
        if feature in categorical:
            label_encoders[feature] = LabelEncoder()
            label_encoders[feature].fit(df[feature])
            
            encoded_df[feature] = label_encoders[feature].transform(df[feature])
    return encoded_df, label_encoders

def label_encode_data(df, feature_names, categorical, label_encoders):
    """
    Label encodes categorical features in a DataFrame using existing LabelEncoders.
    
    Parameters:
    - df: The input DataFrame.
    - feature_names: List of feature names of the dataframe.
    - categorical: List of categorical feature names to be encoded.
    - label_encoders: Dictionary of LabelEncoders used for each feature.
    
    Returns:
    - A new DataFrame with label-encoded categorical features."""
    is_series = isinstance(df, pd.Series)
    encoded_df = df.to_frame().T.copy() if is_series else df.copy()

    for feature in feature_names:
        if feature in categorical:
            encoded_df[feature] = label_encoders[feature].transform(df[feature])
    return encoded_df

def decode_label_encoded_data(data, feature_names, categorical_columns, label_encoders):
    """Decodes label-encoded categorical features for pandas DataFrames, NumPy arrays, and lists.
    
    Parameters:
    - data: The input data (can be list, NumPy array, or DataFrame).
    - feature_names: List of feature names of the dataframe.
    - categorical_columns: List of categorical feature names to be decoded.
    - label_encoders: Dictionary of LabelEncoders used for each feature.
    
    Returns:
    - A new DataFrame or NumPy array with decoded categorical features.
    """
    if isinstance(data, list):
        data = np.array(data)

    if isinstance(data, pd.DataFrame):
        decoded_data = data.copy()
        for feature in categorical_columns:
            if feature in feature_names:
                decoded_data[feature] = label_encoders[feature].inverse_transform(data[feature].astype(int))
        return decoded_data
    
    elif isinstance(data, np.ndarray):
        decoded_data = data.copy()
        decoded_dict = {}
        
        for i, feature in enumerate(feature_names):
            if feature in categorical_columns:
                decoded_dict[feature] = label_encoders[feature].inverse_transform(data[:, i].astype(int))
            else:
                decoded_dict[feature] = data[:, i]
        
        decoded_data = np.column_stack([decoded_dict[feature] for feature in feature_names])
        return decoded_data
    
    else:
        raise ValueError("Unsupported input type. Expected list, pandas DataFrame, or NumPy array.")

def required_attributes(dataset):
    '''
    Returns the required attributes, such as the MinMaxScaler for each column, the feature ranges, and the feature types.
    '''
    min_max_scaler_per_column = {}
    features_ranges = {}
    feature_types = {}
    for col in dataset.columns:
        min_max_scaler_per_column[col] = MinMaxScaler().fit(dataset[col].values.reshape(-1, 1))
        minimum = min_max_scaler_per_column[col].data_min_[0]
        maximum = min_max_scaler_per_column[col].data_max_[0]
        features_ranges[col] = (minimum, maximum)
        if pd.api.types.is_integer_dtype(dataset[col]):
            feature_types[col] = 'int'
        elif pd.api.types.is_float_dtype(dataset[col]):
            feature_types[col] = 'float'
    
    return min_max_scaler_per_column, features_ranges, feature_types
############################################################################################################################
############################################################################################################################
############################################################################################################################
def safe_divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else None

def transform_individual(individual, scaler):
    return scaler.transform(individual.reshape(1, -1))

def inverse_transform_individual(scaled_individual, scaler, feature_columns):
    if len(scaled_individual) != len(feature_columns):
        raise ValueError(f"Expected feature length: {len(feature_columns)}, got: {len(scaled_individual)}. Please check your feature selection pipeline.")
    original_individual = scaler.inverse_transform(scaled_individual.reshape(1, -1))
    return original_individual[0], pd.DataFrame(original_individual, columns=feature_columns).iloc[0].to_dict()

def f_model(instance, model):
    # model is a pipeline
    if isinstance(model, Pipeline):
        if isinstance(instance, pd.DataFrame):
            prediction = model.predict(instance)
        else:
            raise ValueError("Unsupported input type. Expected pandas DataFrame.")
    else:
        if isinstance(instance, pd.DataFrame):
            instance = instance.to_numpy().reshape(1, -1)
        else:
            instance = instance.reshape(1, -1)
        
        prediction = model.predict(instance)[0]

    return prediction

def display_cfe_comparison(original, cfe):
    table = PrettyTable()
    table.field_names = ["Feature", "Original Value", "Proposed CFE", "Change"]

    for feature, original_value in original.items():
        proposed_value = cfe[feature]
        if isinstance(original_value, (int, float)) and isinstance(proposed_value, (int, float)):
            change = f"{proposed_value - original_value:+}" if original_value != proposed_value else ""
        else:
            change = "->" if original_value != proposed_value else ""

        table.add_row([feature, original_value, proposed_value, change])

    table.align = "l"
    print("\nComparison of Initial Instance and Proposed CFE:")
    print(table)