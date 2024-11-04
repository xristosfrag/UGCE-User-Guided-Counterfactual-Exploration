from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from prettytable import PrettyTable

# transform a single individual from the original space to the scaled space
def transform_individual(individual, scaler):
    return scaler.transform(individual.reshape(1, -1))

# Inverse transform a scaled individual back to the original space
def inverse_transform_individual(scaled_individual, scaler, feature_columns):
    if len(scaled_individual) != len(feature_columns):
        raise ValueError(f"Expected feature length: {len(feature_columns)}, got: {len(scaled_individual)}. Please check your feature selection pipeline.")
    original_individual = scaler.inverse_transform(scaled_individual.reshape(1, -1))
    return pd.DataFrame(original_individual, columns=feature_columns).iloc[0].to_dict()

# Scale user-provided constraints from original space to scaled space
def scale_constraints(original_constraints, scaler, feature_columns):
    # Create a dummy dataframe with the constraints
    constraint_df = pd.DataFrame([original_constraints], columns=feature_columns)
    # Scale the constraints using the same scaler
    scaled_constraints = scaler.transform(constraint_df)
    return {i: (scaled_constraints[0][i], scaled_constraints[0][i]) for i in range(len(feature_columns))}

# Train a binary classification model on the COMPAS dataset
def train_compas_model(features, target):
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