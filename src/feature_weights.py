import logging
import numpy as np
from utils import normalize_data

def get_mads(dataset, feature_names, numerical_columns, normalized=False):
    """Computes Median Absolute Deviation of features."""
    mads = {}
    if normalized is False:
        for feature in numerical_columns:
            mads[feature] = np.median(
                abs(dataset[feature].values - np.median(dataset[feature].values)))
    else:
        normalized_train_df = normalize_data(dataset, feature_names, dataset)
        for feature in numerical_columns:
            mads[feature] = np.median(
                abs(normalized_train_df[feature].values - np.median(normalized_train_df[feature].values)))
    return mads

def get_valid_mads(dataset, feature_names, numerical_columns, normalized=False, display_warnings=False, return_mads=True):
    """Computes Median Absolute Deviation of features. If they are <=0, returns a practical value instead"""
    mads = get_mads(dataset, feature_names, numerical_columns, normalized=normalized)
    for feature in mads:
        if mads[feature] <= 0:
            mads[feature] = 1.0
            if display_warnings:
                logging.warning(" MAD for feature %s is 0, so replacing it with 1.0 to avoid error.", feature)

    if return_mads:
        return mads

def get_feature_weights(dataset, feature_names, numerical_columns, features_ranges, feature_weights, encoding='one-hot'):
    """
    Initializes variables related to the main loss function, including feature weights based on different strategies.

    Parameters:
    - dataset: Pandas DataFrame containing the dataset.
    - numerical_columns: List of numerical feature names.
    - feature_weights: Method for calculating feature weights ('inverse_mad', 'label', or custom dictionary).
    - encoding: Type of encoding ('one-hot' or 'label').

    Returns:
    - feature_weights_list: List of feature weights.
    """
    feature_weights_input = None
    feature_weights_list = []

    if feature_weights != feature_weights_input:
        feature_weights_input = feature_weights

        if feature_weights == "inverse_mad":
            normalized_mads = get_valid_mads(dataset, feature_names, numerical_columns, normalized=False)
            feature_weights = {}
            for feature in normalized_mads:
                feature_weights[feature] = round(1 / normalized_mads[feature], 2)
        
        feature_weights_list = []
        # Determine weights based on encoding type
        if encoding == 'one-hot':
            for feature in dataset.columns:
                feature_weights_list.append(feature_weights.get(feature, 1.0))  # Default weight 1.0 if missing
        
        elif encoding == 'label':
            for feature in dataset.columns:
                if feature in feature_weights:
                    feature_weights_list.append(feature_weights[feature])
                else:
                    feature_weights_list.append(round(1 / float(features_ranges[feature][1]), 2))

        feature_weights_list = [feature_weights_list]
    return feature_weights_list