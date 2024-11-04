import sys
import os

def get_Dynamic_CFEs_directory():
    """Get the path of the 'DynamicCFEs' directory."""
    current_dir = os.getcwd()
    target_dir = 'DynamicCFEs'
    
    while os.path.basename(current_dir) != target_dir:
        current_dir = os.path.dirname(current_dir)
        if current_dir == os.path.dirname(current_dir):
            return None
        
    return current_dir

def get_system_slash():
    """Get the system-specific directory separator."""
    return os.sep

Dynamic_CFEs = get_Dynamic_CFEs_directory()
sys.path.append(Dynamic_CFEs)
sep = get_system_slash()
sys.path.append(Dynamic_CFEs + get_system_slash() + 'src')


def main():
    # Define the problem using DEAP's creator
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    # Load COMPAS data
    features, target, scaler, feature_columns, categorical_columns, numerical_columns, one_hot_encode_features, feature_scales = load_compas()
    # features = features.to_numpy()
    # target = target.to_numpy()

    # Train a logistic regression model
    model, X_train, X_test, y_train, y_test = train_compas_model(features, target)

    # Find the first negative instance from the model predictions on the test set
    negative_instances = X_test[model.predict(X_test) == 0]
    x = negative_instances[0]  # First negative instance

    # User-defined model function
    def f_model(instance, model):
        return int(model.predict(instance.reshape(1, -1))[0])


    # Initialize constraints based on 'data_distribution'
    immutables = {'sex', 'race'}  # Example: 'sex' and 'race' are immutable in COMPAS
    immutables = handle_immutable_onehot(immutables, one_hot_encode_features, feature_columns)
    # Use 'data_distribution' option to initialize constraints from feature scales
    constraints = initialize_constraints(feature_columns, feature_scales, data_distribution=True)

    # Define evolutionary parameters
    N = 50  # Population size
    n = x.shape[0]  # Number of features
    lambda1 = 1.0  # Sparsity weight
    lambda2 = 10.0  # Constraint violation weight

    # Create toolbox for DEAP evolutionary algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_individual, n=n, constraints=constraints, immutables=immutables, x=x, categorical_values=feature_scales, numerical_ranges=feature_scales, feature_columns=feature_columns)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    # Register the bounded mutation function with partial, passing constraints
    toolbox.register("mutate", partial(bounded_mutation, n=n, categorical_values=feature_scales, numerical_ranges=feature_scales, constraints=constraints))
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run the evolutionary process to find counterfactuals
    final_population = evolve(x, model=model, initial_constraints=constraints, lambda1=lambda1, lambda2=lambda2, scaler=scaler, feature_columns=feature_columns)
    return final_population

if __name__ == '__main__':
    main()