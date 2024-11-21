import pandas as pd
import numpy as np
from aif360.sklearn.datasets import fetch_compas
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer

import sys
import os

def get_UGCE_directory():
    """Get the path of the 'UGCE-User-Guided-Counterfactual-Exploration' directory."""
    current_dir = os.getcwd()
    target_dir = 'UGCE-User-Guided-Counterfactual-Exploration'
    
    while os.path.basename(current_dir) != target_dir:
        current_dir = os.path.dirname(current_dir)
        if current_dir == os.path.dirname(current_dir):
            return None
        
    return current_dir

def get_system_slash():
    """Get the system-specific directory separator."""
    return os.sep

ugce_dir = get_UGCE_directory()
sys.path.append(ugce_dir)
sep = get_system_slash()
sys.path.append(ugce_dir + get_system_slash() + 'src')

def calculate_num_bins(num_unique_values, value_range):
    num_bins = min(6, int(np.log2(num_unique_values)) + 1)
    num_bins = min(num_bins, value_range)
    return num_bins

def preprocess_dataset(df, continuous_features=[], one_hot_encode=True, datasetName="Adult"):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder()
    numeric_columns = []
    categorical_columns = []
    one_hot_encode_features = []
    feature_types = {}  # Dictionary to store whether features are int or float

    for col in df.columns:
        if (df[col].dtype == 'object' or df[col].dtype == 'category') and col not in continuous_features:
            categorical_columns.append(col)
            if len(df[col].unique()) == 2 or (datasetName == 'GermanCredit' and col in ['Existing-Account-Status', 'Savings-Account', 'Guarantors', 'Installment', 'Job', 'Property', 'Housing', 'Present-Employment']):
                df[col] = label_encoder.fit_transform(df[col])
            elif one_hot_encode or (one_hot_encode and datasetName == "Adult" and col == 'race'):
                encoded_values = onehot_encoder.fit_transform(df[[col]])
                new_cols = [col + '_' + str(i) for i in range(encoded_values.shape[1])]
                encoded_df = pd.DataFrame(encoded_values.toarray(), columns=new_cols)
                df = pd.concat([df, encoded_df], axis=1)
                df.drop(col, axis=1, inplace=True)
                one_hot_encode_features.extend(new_cols)
        elif (df[col].dtype == 'object' or df[col].dtype == 'category') and df[col].str.isnumeric().all() and col not in continuous_features:
            df[col] = df[col].astype(int)
            categorical_columns.append(col)
        elif col in continuous_features:
            if pd.api.types.is_integer_dtype(df[col]):
                    feature_types[col] = 'int'
            elif pd.api.types.is_float_dtype(df[col]):
                feature_types[col] = 'float'
            
            numeric_columns.append(col)
            num_unique_values = len(df[col].unique())
            value_range = df[col].max() - df[col].min()
            num_bins = calculate_num_bins(num_unique_values, value_range)
            bin_discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform', subsample=None)
            bins = bin_discretizer.fit_transform(df[[col]])
            df[col] = bins.astype(int)
        else:
            if len(df[col].unique()) > 2:
                numeric_columns.append(col)
            
                if pd.api.types.is_integer_dtype(df[col]):
                    feature_types[col] = 'int'
                elif pd.api.types.is_float_dtype(df[col]):
                    feature_types[col] = 'float'
    return df, numeric_columns, categorical_columns, one_hot_encode_features, feature_types

def load_compas():
    X, y = fetch_compas()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    TARGET_COLUMNS = 'two_year_recid'
    data = X
    data = data.drop(['c_charge_desc', 'age_cat'], axis=1)
    data, numeric_columns, categorical_columns, one_hot_encode_features, feature_types = preprocess_dataset(data, continuous_features=[])
    data_df_copy = data.copy()
    y = pd.DataFrame(y, columns=[TARGET_COLUMNS])
    FEATURE_COLUMNS = data.columns
    y, _, _, _, _ = preprocess_dataset(y, continuous_features=[])
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data[FEATURE_COLUMNS].values)
    features_ranges = {}
    for col in data.columns:
        if col in one_hot_encode_features or col in categorical_columns:
            # For categorical and one-hot encoded features, return unique values as native Python types
            features_ranges[col] = [int(value) if isinstance(value, np.integer) else value for value in data[col].unique()]
        else:
            # For numerical features, return the min and max as native Python types
            min_val = data[col].min()
            max_val = data[col].max()
            features_ranges[col] = (int(min_val), int(max_val)) if feature_types[col] == 'int' else (float(min_val), float(max_val))
    
    # Convert the data to list format instead of NumPy array
    return (
        data_scaled,  # Convert features to list
        y.values,   # Convert target to list
        min_max_scaler,
        FEATURE_COLUMNS,
        categorical_columns,
        numeric_columns,
        one_hot_encode_features,
        features_ranges,
        feature_types
    )