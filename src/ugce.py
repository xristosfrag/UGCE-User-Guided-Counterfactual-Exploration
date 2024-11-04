import random
import numpy as np
from functools import partial
from collections import Counter
from copy import deepcopy
from time import time

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

from utils import transform_individual, inverse_transform_individual, f_model, display_cfe_comparison

random.seed(42)
np.random.seed(42)

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None
        
    def set_fittness(self, fitness):
        self.fitness = fitness
        
    def __repr__(self):
        return f"Individual({self.genes}, fitness={self.fitness})"

class UGCE:
    def __init__(self, model, scaler, feature_columns, categorical_columns,\
        numerical_columns, one_hot_encode_features, features_ranges, features_type):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.one_hot_encode_features = one_hot_encode_features
        self.features_ranges = features_ranges
        self.features_type = features_type
        self.seed_update_number = 0
        
        
    def set_seed(self, seed_number):
        """Set seeds for reproducibility in random and np.random."""
        random.seed(seed_number)
        np.random.seed(seed_number)

    def tournament_selection(self, population, k):
        winners = []
        for _ in range(k):
            participants = random.sample(population, self.tournsize)
            winners.append(max(participants, key=lambda ind: ind.fitness))
        return winners

    def roulette_selection(self, population, k):
        total_fitness = sum(ind.fitness for ind in population)
        selected = []
        for _ in range(k):
            pick = random.uniform(0, total_fitness)
            current = 0
            for ind in population:
                current += ind.fitness
                if current > pick:
                    selected.append(ind)
                    break
        return selected

    def rank_selection(self, population, k):
        population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        return population[:k]

    def sus_selection(self, population, k):
        total_fitness = sum(ind.fitness for ind in population)
        distance = total_fitness / k
        start_point = random.uniform(0, distance)
        selected = []
        for i in range(k):
            current = start_point + i * distance
            cumulative_fitness = 0
            for ind in population:
                cumulative_fitness += ind.fitness
                if cumulative_fitness >= current:
                    selected.append(ind)
                    break
        return selected

    def explain_instance(self, x, constraints=None, immutables=None,\
        diversity_top_k=1, evaluation=False, dynamic_constraints=False,\
        initial_population_variability=0.2, data_distribution=True,\
            seed_number=42,\
            num_generations=50, population_size=50, num_parents=10,\
            selection_method="tournament", tournsize=3,\
            early_stopping_iterations=3, elite_ratio=0.1, \
             lamda1=1, lamda2=1, lamda3=1, lamda4=1, cxpb=0.5, crossover_points=3, mutpb=0.2):
        """
        Explain the instance by evolving counterfactual examples.
        """           
        self.x = x
        ## get the scaled individual for the original instance x to use it as a reference for the new individual
        self.inverse_transformed_x = inverse_transform_individual(self.x, self.scaler, self.feature_columns)
        print(f"Explaining instance {self.inverse_transformed_x}")
        self.diversity_top_k = diversity_top_k
        self.evaluation = evaluation
        self.dynamic_constraints = dynamic_constraints
        self.initial_population_variability = initial_population_variability
        self.num_generations = num_generations
        self.early_stopping_iterations = early_stopping_iterations
        self.elite_ratio = elite_ratio # Percentage of individuals to retain from both current and offspring population
        self.initial_population_variability = initial_population_variability
        self.data_distribution = data_distribution
        
        self.seed_number = seed_number
        self.seed_update_number = 0
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents = num_parents
        self.selection_method = selection_method
        self.tournsize = tournsize
        
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.lamda3 = lamda3
        self.lamda4 = lamda4
        self.cxpb = cxpb
        self.crossover_points = crossover_points
        self.mutpb = mutpb
        # Reset seeds to ensure reproducibility in each call
        self.set_seed(self.seed_number)
        
        self.original_prediction = f_model(x, self.model)
        self.constraints = constraints if constraints else {}
        self.immutables = immutables if immutables else []
        self.setup_constraints()
        best_individuals = self.evolve()
        print(f"Best cfe is: {best_individuals.genes}, guaranteed to alter the decision of the model from {self.original_prediction} to: {f_model(best_individuals.genes, self.model)}")
        return best_individuals

    def distance(self, x_prime):
        return np.linalg.norm(np.array(self.x) - np.array(x_prime))

    def sparsity(self, x_prime):
        return np.sum(np.abs(np.array(self.x) - np.array(x_prime)) > 1e-6)

    def violation(self, x_prime, verbose=False):
        """
        Calculate the violation of the constraints for the new solution x_prime.
        The constraints are given as a dictionary with the key being the index of the feature and the value being a tuple
        with the lower and upper bounds of the feature.
        The immutables are a set of indices of features that are immutable.
        The original_x is the explainee instance.
        """
        penalty = 0
        reward = 0
        
        for attribute_index, value in enumerate(x_prime):
            if attribute_index in self.immutables:
                # Feature is immutable, reward more if unchanged, penalize more if changed
                if x_prime[attribute_index] == self.x[attribute_index]:
                    reward += 30  # Larger reward for immutable feature remaining unchanged
                    if verbose:
                        print("Feature {} is immutable and unchanged. Reward: {}".format(attribute_index, reward))
                else:
                    penalty += 30  # Larger penalty for immutable feature changing
                    if verbose:
                        print("Feature {} is immutable and changed. Penalty: {}".format(attribute_index, penalty))
            else:
                if attribute_index not in self.constraints:
                    # No constraints on this feature
                    continue
                lower, upper = self.constraints[attribute_index]
                if lower <= x_prime[attribute_index] <= upper:
                    # Reward if the feature is within the bounds
                    reward += 30
                    if verbose:
                        print("Feature {} is within bounds. Reward: {}".format(attribute_index, reward))
                else:
                    # Apply penalty if the feature is outside the bounds
                    penalty += abs(x_prime[attribute_index] - lower) if x_prime[attribute_index] < lower else abs(x_prime[attribute_index] - upper)
                    if verbose:
                        print("Feature {} is outside bounds. Penalty: {}".format(attribute_index, penalty))

        # Return both the penalty and the reward
        return penalty, reward
   
    def evaluate(self, individual):            
        y_prime = f_model(transform_individual(np.array(individual), self.scaler), self.model)
        d = self.distance(individual)
        s = self.sparsity(individual)
        penalty, reward = self.violation(individual)
        if y_prime == self.original_prediction:
            penalty += 10000
        else:
            reward += 1000
        return - self.lamda1 * d - self.lamda2 * s - self.lamda3 * penalty + self.lamda4 * reward
    
    """
    # Minimum and maximum value for the evaluation function
    # Minimum: All features are the same as the original instance, within the constraints
    # Maximum: All features are different from the original instance, within the constraints

    # d_max = sqrt(sum((x_i - x'_i)^2)) for all i = sqrt(len(x))
    # s_max = len(x)
    # v_max = sum(abs(x_i - x'_i)) for all i
    # penalty_max = 100
    # reward_max = 5 * len(immutables)

    # Evaluation function:
    Minimum value: -d_max
    # """

    def generate_individual(self):
        """
        Generate an individual by considering constraints, immutability, and categorical/numerical ranges.
        :return: An individual
        """        
        genes = []
        skip_indices = set()  # To keep track of one-hot encoded features to skip

        for i in range(len(self.x)):
            skip_feature_change_prob = random.random()

            if i in skip_indices:
                continue
            feature_name = self.feature_columns[i]
            
            # If the feature is immutable, keep its original value
            if i in self.immutables:
                genes.append(self.inverse_transformed_x[feature_name])
                continue
            
            # For categorical features, randomly select one of the known unique values
            elif feature_name in self.categorical_columns:
                if self.data_distribution:
                    possible_values = self.features_ranges[feature_name]
                    genes.append(random.choice(possible_values))
                else:
                    # If the data distribution is not known, generate a random integer
                    genes.append(random.randint(0, len(self.features_ranges[feature_name]) - 1))
                

            # For one-hot encoded features, randomly select one of the features in the group to set to 1
            elif feature_name in self.one_hot_encode_features:
                one_hot_group = [f for f in self.one_hot_encode_features if f.startswith(feature_name.split('_')[0])]
                ## if skip the feature change then keep the original values for the entire group
                if skip_feature_change_prob > self.initial_population_variability:
                    # Keep original values for the entire group
                    genes.extend([self.inverse_transformed_x[f] for f in one_hot_group])
                    skip_indices.update([list(self.feature_columns).index(f) for f in one_hot_group])
                    continue
                
                chosen_feature = random.choice(one_hot_group)
                # Set the chosen feature to 1 and all others in the group to 0
                for one_hot_feature in one_hot_group:
                    if one_hot_feature == chosen_feature:
                        genes.append(1)
                    else:
                        genes.append(0)

                # Skip further iterations for the one-hot encoded features in this group
                skip_indices.update([list(self.feature_columns).index(f) for f in one_hot_group])
            
            # For numerical features, generate values within the defined constraints (if provided) or range
            else:
                # Randomly decide whether to change this feature or not
                if skip_feature_change_prob > self.initial_population_variability:
                    genes.append(self.inverse_transformed_x[feature_name])
                    continue
                
                if self.constraints.get(i):  # Check if specific constraints are provided
                    lower, upper = self.constraints[i]

                    ## check if the constrains violate the data distribution of the feature
                    if self.data_distribution:
                        lower_data_distribution, upper_data_distribution = self.features_ranges[feature_name]

                        if lower < lower_data_distribution or upper > upper_data_distribution:
                            print(f"Constraints for {feature_name} violate the data distribution: [{lower}, {upper}] vs [{lower_data_distribution}, {upper_data_distribution}]")
                            sys.exit()
                else:
                    # If no constraints are provided, use the known range for this feature
                    lower, upper = self.features_ranges[feature_name]

                if self.features_type[feature_name] == 'int':
                    genes.append(random.randint(lower, upper))
                else:
                    genes.append(random.uniform(lower, upper))

        return Individual(genes)
        
    def crossover(self, parents, crossover_points=1):
        num_parents = len(parents)
        num_features = len(self.feature_columns)
        crossover_points = sorted(random.sample(range(1, num_features), crossover_points))

        # Initialize two offspring individuals
        offspring1 = np.zeros(num_features)
        offspring2 = np.zeros(num_features)

        skip_indices = set()
        current_parent_idx = random.randint(0, num_parents - 1)
        start_idx = 0

        for point in crossover_points:
            for i in range(start_idx, point):
                feature_name = self.feature_columns[i]
                if feature_name in self.one_hot_encode_features:
                    if i in skip_indices:
                        continue
                    one_hot_group = [f for f in self.one_hot_encode_features if f.startswith(feature_name.split('_')[0])]
                    chosen_feature1 = next(idx for idx in one_hot_group if parents[current_parent_idx].genes[list(self.feature_columns).index(idx)] == 1)
                    chosen_feature2 = next(idx for idx in one_hot_group if parents[(current_parent_idx + 1) % num_parents].genes[list(self.feature_columns).index(idx)] == 1)

                    for one_hot_feature in one_hot_group:
                        index = list(self.feature_columns).index(one_hot_feature)
                        offspring1[index] = 1 if one_hot_feature == chosen_feature1 else 0
                        offspring2[index] = 1 if one_hot_feature == chosen_feature2 else 0
                    skip_indices.update([list(self.feature_columns).index(f) for f in one_hot_group])
                else:
                    offspring1[i] = parents[current_parent_idx].genes[i]
                    offspring2[i] = parents[(current_parent_idx + 1) % num_parents].genes[i]
            current_parent_idx = (current_parent_idx + 1) % num_parents
            start_idx = point

        for i in range(start_idx, num_features):
            feature_name = self.feature_columns[i]
            if feature_name in self.one_hot_encode_features:
                if i in skip_indices:
                    continue
                one_hot_group = [f for f in self.one_hot_encode_features if f.startswith(feature_name.split('_')[0])]
                chosen_feature1 = next(idx for idx in one_hot_group if parents[current_parent_idx].genes[list(self.feature_columns).index(idx)] == 1)
                chosen_feature2 = next(idx for idx in one_hot_group if parents[(current_parent_idx + 1) % num_parents].genes[list(self.feature_columns).index(idx)] == 1)

                for one_hot_feature in one_hot_group:
                    index = list(self.feature_columns).index(one_hot_feature)
                    offspring1[index] = 1 if one_hot_feature == chosen_feature1 else 0
                    offspring2[index] = 1 if one_hot_feature == chosen_feature2 else 0
                skip_indices.update([list(self.feature_columns).index(f) for f in one_hot_group])
            else:
                offspring1[i] = parents[current_parent_idx].genes[i]
                offspring2[i] = parents[(current_parent_idx + 1) % num_parents].genes[i]

        return Individual(offspring1), Individual(offspring2)

    def mutate_individual(self, individual):
        for i in range(len(self.feature_columns)):
            self.seed_update_number += 1
            self.set_seed(self.seed_number + self.seed_update_number)
            # print(f"Mutation per attribute Seed number: {self.seed_number + self.seed_update_number}")
            if random.random() < self.mutpb:
                feature_name = self.feature_columns[i]
                if i in self.immutables:
                    continue
                if feature_name in self.categorical_columns:
                    possible_values = self.features_ranges[feature_name]
                    original_value = individual.genes[i]
                    new_value = original_value
                    while new_value == original_value:
                        new_value = random.choice(possible_values)
                    individual.genes[i] = new_value
                elif feature_name in self.one_hot_encode_features:
                    one_hot_group = [f for f in self.one_hot_encode_features if f.startswith(feature_name.split('_')[0])]
                    current_index = next(idx for idx in one_hot_group if individual.genes[list(self.feature_columns).index(idx)] == 1)
                    chosen_feature = current_index
                    while chosen_feature == current_index:
                        chosen_feature = random.choice(one_hot_group)
                    for one_hot_feature in one_hot_group:
                        index = list(self.feature_columns).index(one_hot_feature)
                        individual.genes[index] = 1 if one_hot_feature == chosen_feature else 0
                else:
                    original_value = individual.genes[i]
                    new_value = original_value
                    if self.constraints.get(i):
                        lower, upper = self.constraints[i]
                        if self.data_distribution:
                            lower_data_distribution, upper_data_distribution = self.features_ranges[feature_name]
                            if lower < lower_data_distribution or upper > upper_data_distribution:
                                print(f"Constraints for {feature_name} violate the data distribution: [{lower}, {upper}] vs [{lower_data_distribution}, {upper_data_distribution}]")
                                sys.exit()
                    else:
                        lower, upper = self.features_ranges[feature_name]
                    if self.features_type[feature_name] == 'int':
                        while new_value == original_value:
                            new_value = random.randint(lower, upper)
                    else:
                        while new_value == original_value:
                            new_value = random.uniform(lower, upper)
                    individual.genes[i] = new_value
        return individual
    
    def fitness_assignment(self, population, clear_fitness=False):
        """
        Assign fitness values to the population based on the evaluation function.
        
        Args:
            population (list): The population of individuals to evaluate.
        """
        self.set_seed(self.seed_number)
        if clear_fitness:
            for ind in population: # recalculate the fitness for the entire population
                ind.fitness = self.evaluate(ind.genes)
        else:
        for ind in population:
            ind.fitness = ind.fitness if ind.fitness is not None else self.evaluate(ind.genes)
            
    def population(self):
        population = []
        unique_individuals = set()
        while len(population) < self.population_size:
            new_individual = self.generate_individual()
            genes_tuple = tuple(new_individual.genes)
            if genes_tuple not in unique_individuals:
                population.append(new_individual)
                unique_individuals.add(genes_tuple)
        return population

    def evolve(self):
        # print("Initial seed number: ", self.seed_number + self.seed_update_number)
        population = self.population()
        print(f"    Diversity: {100 - self.identical_individuals_percentage(population):.2f}% unique individuals")
        
        self.fitness_assignment(population)        
        print(f"Constraints: {self.constraints}")
        print(f"Immutable features: {self.immutables}")
             
        max_fitness, avg_fitness = self.max_avg_fitness(population)
        print(f"Initial population average fitness: {avg_fitness}, max fitness: {max_fitness}")
        best_fitness = float("-inf")
        
        def generations(population, num_generations, best_fitness):
            print("Starting evolution...")
            # Initialize variables to track improvements
            generations_without_improvement = 0
            elite_count = max(1, int(self.elite_ratio * len(population)))  # Calculate the number of elite individuals

            for gen in range(num_generations):
                # Termination criterion based on lack of improvement
                if generations_without_improvement >= self.early_stopping_iterations:
                    print("Stopping early due to lack of fitness improvement.")
                    break
                # Re-seed at the start of each generation
                self.seed_update_number += gen
                self.set_seed(self.seed_number + self.seed_update_number)

                print(f"Generation: {gen}")
                if self.elite_ratio > 0:
                    # Select the top elite individuals from the current population
                    elite_individuals = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:elite_count]

                # print(f"Generation Seed number: {self.seed_number + self.seed_update_number}")
                if self.selection_method == "tournament":
                    parents = self.tournament_selection(population, self.num_parents)
                elif self.selection_method == "roulette":
                    parents = self.roulette_selection(population, self.num_parents)
                elif self.selection_method == "rank":
                    parents = self.rank_selection(population, self.num_parents)
                elif self.selection_method == "sus":
                    parents = self.sus_selection(population, self.num_parents)
                
                offspring = []
                offspring_size = len(population) - elite_count

                # Create offspring until the required population size is reached
                while len(offspring) < offspring_size:
                    self.seed_update_number += 1
                    self.set_seed(self.seed_number + self.seed_update_number)
                    # print(f"Crossover Seed number: {self.seed_number + self.seed_update_number}")

                    # Select parent pairs for crossover, allowing repetition of parents
                    parent1, parent2 = deepcopy(random.choice(parents)), deepcopy(random.choice(parents))
                    if random.random() < self.cxpb:
                        # Perform crossover
                        child1, child2 = self.crossover([parent1, parent2])
                        offspring.extend([child1, child2])
                    else:
                        # If no crossover, directly clone parents
                        offspring.extend([parent1, parent2])

                for mutant in offspring:
                    self.seed_update_number += 1  
                    self.set_seed(self.seed_number + self.seed_update_number)
                    # print(f"Mutation Seed number: {self.seed_number + self.seed_update_number}")

                    if random.random() < self.mutpb:
                        self.mutate_individual(mutant)
                
                # Evaluate the fitness of the offspring
                self.fitness_assignment(offspring)
                print(f"Population size: {len(population)}, offspring size: {len(offspring)}")

                # Ensure only the needed number of offspring are retained
                if self.elite_ratio > 0:
                    ## keep the best individuals from the offspring (offspring_size)
                    offspring = sorted(offspring, key=lambda ind: ind.fitness, reverse=True)[:offspring_size]
                    
                    # Combine elite individuals with the new offspring to form the next generation
                    population = elite_individuals + offspring
                else:
                    ## keep the best individuals from the current population and the offspring
                    population = sorted(population + offspring, key=lambda ind: ind.fitness, reverse=True)[:self.population_size]

                print("Final population size: ", len(population))
                best_individual = max(population, key=lambda ind: ind.fitness)
                print(f"Generation {gen}: Best fitness {best_individual.fitness}")
                    
                current_best_fitness, avg_fitness = self.max_avg_fitness(population)
                print(f"    Average fitness: {avg_fitness}, max fitness: {current_best_fitness}")
                # Track improvement
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
            print("\nEvolution complete.")
            return population
        
        population = generations(population, self.num_generations, best_fitness)
        print(f"    Diversity: {100 - self.identical_individuals_percentage(population):.2f}% unique individuals")

        best_individuals = self.best_individuals(population, self.diversity_top_k)
        cfe_with_feature_names = dict(zip(self.feature_columns, best_individuals.genes))  # Transform the best individual list to a dictionary format
        display_cfe_comparison(self.inverse_transformed_x, cfe_with_feature_names)
        print()
    
        # If dynamic constraints are enabled, ask the user for acceptance and update constraints
        if self.dynamic_constraints:
            print("Dynamic constraints enabled.")
            best_individuals = self.best_individuals(population, self.diversity_top_k)
            cfe_with_feature_names = dict(zip(self.feature_columns, best_individuals.genes))  # Transform the best individual list to a dictionary format
            accepted = self.ask_user_acceptance()            
            
            while not accepted:
                # Update constraints based on user input
                self.get_updated_constraints()
                
                # Update the fitness values based on the new constraints
                self.fitness_assignment(population, clear_fitness=True)
                
                print(f"Constraints: {self.constraints}")
                print(f"Immutable features: {self.immutables}")
                population = generations(population, 30, max_fitness)
                print(f"    Diversity: {100 - self.identical_individuals_percentage(population):.2f}% unique individuals")
                best_individuals = self.best_individuals(population, self.diversity_top_k)
                cfe_with_feature_names = dict(zip(self.feature_columns, best_individuals.genes))  # Transform the best individual list to a dictionary format
                display_cfe_comparison(self.inverse_transformed_x, cfe_with_feature_names)
                print()
                accepted = self.ask_user_acceptance()
            return best_individuals
        else:
            best_individuals = self.best_individuals(population, self.diversity_top_k)
            return best_individuals
        
    def identical_individuals_percentage(self, population):
        """
        Calculate the percentage of identical individuals in the population.
        
        ### Inputs:
        - population: List of Individual instances in the population
        
        ### Output:
        - score: Percentage of identical individuals in the population
        
        ### Interpretation:
        - A higher percentage of identical individuals indicates a lack of diversity in the population.
        - A diverse population is more likely to contain a wider range of solutions.
        """
        # Convert each individual's genes to a tuple so they can be counted
        individual_tuples = [tuple(ind.genes) for ind in population]
        # Count occurrences of each unique individual
        counts = Counter(individual_tuples)
        # Calculate the number of identical individuals based on their occurrences
        identical_count = sum(count for count in counts.values() if count > 1)
        # Calculate the percentage of identical individuals in the population
        score = (identical_count / len(population)) * 100
        return score

    def best_individuals(self, population, n=1):
        return max(population, key=lambda ind: ind.fitness)
    
    def max_avg_fitness(self, population):
        fittness_values = [ind.fitness for ind in population]
        return max(fittness_values), sum(fittness_values) / len(population)
    
    # Helper function to ask user for acceptance, presenting the CFE in the original feature space
    def ask_user_acceptance(self):   
        # Mock user input (replace with actual user input handling)
        while True:
            user_response = input("Do you accept this counterfactual explanation? (y/n): ").strip().lower()
            if user_response in ["y", "n"]:
                return user_response == "y"
            else:
                print("Invalid input. Please type 'y' or 'n'.")

    # Mock function to get updated constraints from the user in the original space
    def get_updated_constraints(self):
        print(f"Updating constraints for features values...")

        # Get base feature name for one-hot encoded features
        one_hot_base_features = set(f.split('_')[0] for f in self.one_hot_encode_features)        
        skip_indices = []
        
        for i, feature in enumerate(self.feature_columns):
            if i in skip_indices:
                continue
            
            base_feature_name = feature.split('_')[0]    
            if base_feature_name in one_hot_base_features:
                while True:
                    user_input = input(f"Set '{base_feature_name}' as immutable? ('i' for immutable, 'ni' to remove immutability or Enter to skip): ").strip()
                    
                    find_indices = [i for i, f in enumerate(self.feature_columns) if f.startswith(base_feature_name)]
                    skip_indices.extend(find_indices)
                    
                    if user_input == "":
                        print(f"Feature '{base_feature_name}' is passed (no new constraints).")
                        break
                    elif user_input == "i":
                        print(f"Feature '{base_feature_name}' marked as immutable.")
                        for idx in find_indices:
                            if idx not in self.immutables:
                                self.immutables.append(idx)
                        break
                    elif user_input == "ni":
                        print(f"Feature '{base_feature_name}' is no longer immutable.")
                        ## remove the immutable constraint
                        ## pop the skip indices from the self.immutables
                        for idx in find_indices:
                            self.immutables.remove(idx)                        
                        break
            else:
                while True:
                    user_input = input(f"Enter the lower and upper bounds for feature '{feature}' (or 'i' to make it immutable or 'ni' to remove immutability, Enter to skip): ").strip()
                    
                    # User presses enter to pass the feature (no change)
                    if user_input == "":
                        print(f"Feature '{feature}' is passed (no new constraints).")
                        break
                    
                    # User enters '-' to mark the feature as immutable
                    elif user_input == "i":
                        print(f"Feature '{feature}' marked as immutable.")
                        self.immutables.append(i)
                        break
                    
                    elif user_input == "ni":
                        print(f"Feature '{feature}' is no longer immutable.")
                        ## remove the immutable constraint
                        self.immutables.remove(i)
                        break
                    
                    # User enters a lower and upper bound
                    else:
                        try:
                            lower, upper = map(float, user_input.split())
                            if lower > upper:
                                print("Lower bound cannot be greater than upper bound. Please try again.")
                            else:
                                self.constraints[i] = (lower, upper)
                                break
                        except ValueError:
                            print("Invalid input. Please enter two numeric values or '-' to mark as immutable.")

    def setup_constraints(self):
        ## transform the constraints to indices
        constraints = {}
        self.immutables = []
        for feature, value in self.constraints.items():
            ## find if the feature is in the first part of the one-hot encoded features before the '_'
            found_onehot_encoded_feature = False
            for f in self.one_hot_encode_features:
                if feature.startswith(f.split('_')[0]):
                    found_onehot_encoded_feature = True
                    self.immutables.append(list(self.feature_columns).index(f))
            
            if found_onehot_encoded_feature:
                continue
            if value == '-':
                self.immutables.append(list(self.feature_columns).index(feature))
            else:
                index = list(self.feature_columns).index(feature)
                constraints[index] = value
        self.constraints = constraints