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
            seed_number=42, fix_population=True, complete_random=True, population_size_dynamic=10,\
            num_generations=50, population_size=50, regeneration_tries=2, num_parents=10,\
            selection_method="tournament", tournsize=3,\
            early_stopping_iterations=3, elite_ratio=0.1, \
             lambda1=1, lambda2=1, lambda3=1, lambda4=1, lambda5=1, cxpb=0.5, crossover_points=3, mutpb=0.2):
        """
        Explain the instance by evolving counterfactual examples.
        """
        self.x = x
        ## get the scaled individual for the original instance x to use it as a reference for the new individual
        self.inverse_transformed_x_indexes, self.inverse_transformed_x_features = inverse_transform_individual(self.x, self.scaler, self.feature_columns)
        print(f"Explaining instance {self.inverse_transformed_x_features}")
        self.diversity_top_k = diversity_top_k
        self.evaluation = evaluation
        self.dynamic_constraints = dynamic_constraints
        self.initial_population_variability = initial_population_variability
        self.num_generations = num_generations
        self.early_stopping_iterations = early_stopping_iterations
        self.elite_ratio = elite_ratio # Percentage of individuals to retain from both current and offspring population
        self.initial_population_variability = initial_population_variability
        self.data_distribution = data_distribution
        
        self.fix_population = fix_population
        self.population_size_dynamic = population_size_dynamic
        self.complete_random = complete_random
        self.num_generations = num_generations
        self.population_size = population_size
        self.regeneration_tries = regeneration_tries
        self.num_parents = num_parents
        self.selection_method = selection_method
        self.tournsize = tournsize
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.cxpb = cxpb
        self.crossover_points = crossover_points
        self.mutpb = mutpb
        if not self.complete_random:
            self.seed_number = seed_number
            self.seed_update_number = 0
            # Reset seeds to ensure reproducibility in each call
            self.set_seed(self.seed_number)
        
        self.original_prediction = f_model(x, self.model)
        self.constraints = constraints if constraints else {}
        self.immutables = immutables if immutables else []
        self.setup_constraints()
        best_individuals = self.evolve()
        print(f"Best cfe is: {best_individuals.genes}, guaranteed to alter the decision of the model from {self.original_prediction} to: {f_model(transform_individual(np.array(best_individuals.genes), self.scaler), self.model)}")
        return best_individuals
    
    def explain_instances(self, X, constraints=None, immutables=None,\
        diversity_top_k=1, evaluation=False, dynamic_constraints=False,\
        initial_population_variability=0.2, data_distribution=True,\
            seed_number=42, fix_population=True, complete_random=True, population_size_dynamic=10,\
            num_generations=50, population_size=50, regeneration_tries=2, num_parents=10,\
            selection_method="tournament", tournsize=3,\
            early_stopping_iterations=3, elite_ratio=0.1, \
                lambda1=1, lambda2=1, lambda3=1, lambda4=1, lambda5=1, cxpb=0.5, crossover_points=3, mutpb=0.2,\
                updated_constraints=None, automatic_user_acceptance=True):
        """
        Explain the instances by evolving counterfactual examples.
        """
        self.X = X
        self.diversity_top_k = diversity_top_k
        self.evaluation = evaluation
        self.dynamic_constraints = dynamic_constraints
        self.initial_population_variability = initial_population_variability
        self.num_generations = num_generations
        self.early_stopping_iterations = early_stopping_iterations
        self.elite_ratio = elite_ratio # Percentage of individuals to retain from both current and offspring population
        self.initial_population_variability = initial_population_variability
        self.data_distribution = data_distribution

        self.fix_population = fix_population
        self.population_size_dynamic = population_size_dynamic
        self.complete_random = complete_random
        self.num_generations = num_generations
        self.population_size = population_size
        self.regeneration_tries = regeneration_tries
        self.num_parents = num_parents
        self.selection_method = selection_method
        self.tournsize = tournsize

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.cxpb = cxpb
        self.crossover_points = crossover_points
        self.mutpb = mutpb
        self.updated_constraints = updated_constraints
        self.automatic_user_acceptance = automatic_user_acceptance
        if not self.complete_random:
            self.seed_number = seed_number
            self.seed_update_number = 0
            # Reset seeds to ensure reproducibility in each call
            self.set_seed(self.seed_number)

        self.original_predictions = {f_model(x, self.model) for x in X}
        self.constraints = constraints if constraints else {}
        self.immutables = immutables if immutables else []
        self.setup_constraints()
        
        results_X = {}
        iteration = 0
        for i, x in enumerate(X):
            results = {'instance': [],\
                    'cfes': [],\
                    'cfes_predictions': [],\
                    'cfes_distances': [],\
                    "unique_applicable_cfes": [],\
                    "elapsed_time": 0}
            
            self.x = x
            ## get the scaled individual for the original instance x to use it as a reference for the new individual
            self.inverse_transformed_x_indexes, self.inverse_transformed_x_features = inverse_transform_individual(self.x, self.scaler, self.feature_columns)
            print(f"Explaining instance {self.inverse_transformed_x_features}")
            self.original_prediction = f_model(x, self.model)
            best_individuals, elapsed_time = self.evolve()

            if len(best_individuals) == 1:
                print(f"Best cfe is: {best_individuals[0].genes}, guaranteed to alter the decision of the model from {self.original_prediction} to: {f_model(transform_individual(np.array(best_individuals[0].genes), self.scaler), self.model)}")
                results['instance'] = self.inverse_transformed_x_features
                results['cfes'] = best_individuals[0].genes
                results['cfes_predictions'] = [f_model(transform_individual(np.array(best_individuals[0].genes), self.scaler), self.model)]
                print([best_individual.genes for best_individual in best_individuals])
                results['cfes_distances'] = [self.distance(best_individuals[0].genes)]
                results['elapsed_time'] = elapsed_time
            else:
                print(best_individuals)
                best_individual = best_individuals[0]
                print(len(best_individuals), best_individual)
                print(f"Best cfe is: {best_individual.genes}, guaranteed to alter the decision of the model from {self.original_prediction} to: {f_model(transform_individual(np.array(best_individual.genes), self.scaler), self.model)}")
                results['instance'] = self.inverse_transformed_x_features
                results['cfes'] = best_individual.genes
                results['cfes_predictions'] = [f_model(transform_individual(np.array(best_individual.genes), self.scaler), self.model) for best_individual in best_individuals]
                print([best_individual.genes for best_individual in best_individuals])
                print(type(self.x), [type(best_individual.genes) for best_individual in best_individuals])
                results['cfes_distances'] = [self.distance(best_individual.genes) for best_individual in best_individuals]
                results['elapsed_time'] = elapsed_time
            
            results_X[i] = results
            iteration += 1
            if iteration == 2:
                break
        return results_X

    def distance(self, x_prime):
        return np.linalg.norm(np.array(self.inverse_transformed_x_indexes) - np.array(x_prime))

    def sparsity(self, x_prime):
        return np.sum(np.abs(np.array(self.inverse_transformed_x_indexes) - np.array(x_prime)) > 1e-6)

    def violation(self, x_prime, verbose=False):
        """
        Calculate the violation of the constraints for the new solution x_prime.
        The constraints are given as a dictionary with the key being the index of the feature and the value being a tuple
        with the lower and upper bounds of the feature.
        The immutables are a set of indices of features that are immutable.
        The original_x is the explainee instance.
        """
        immutable_score = 0
        ranges_score = 0
        
        for attribute_index, value in enumerate(x_prime):
            if attribute_index in self.immutables:
                # Feature is immutable, reward more if unchanged, penalize more if changed
                if x_prime[attribute_index] == self.inverse_transformed_x_indexes[attribute_index]:
                    immutable_score += 1000  # Larger reward for immutable feature remaining unchanged
                    if verbose:
                        print("Feature {} is immutable and unchanged. Reward: {}".format(attribute_index, immutable_score))
                else:
                    immutable_score -= 1000  # Larger penalty for immutable feature changing
                    if verbose:
                        print("Feature {} is immutable and changed. Penalty: {}".format(attribute_index, immutable_score))
            else:
                if attribute_index not in self.constraints:
                    # No constraints on this feature
                    continue
                lower, upper = self.constraints[attribute_index]
                if lower <= x_prime[attribute_index] <= upper:
                    # Reward if the feature is within the bounds
                    ranges_score += 1000
                    if verbose:
                        print("Feature {} is within bounds. Reward: {}".format(attribute_index, ranges_score))
                else:
                    # Apply penalty if the feature is outside the bounds
                    ranges_score -= 1000 * abs(x_prime[attribute_index] - lower) if x_prime[attribute_index] < lower else abs(x_prime[attribute_index] - upper)
                    if verbose:
                        print("Feature {} is outside bounds. Penalty: {}".format(attribute_index, ranges_score))

        # Return both the penalty and the reward
        return immutable_score, ranges_score
   
    def evaluate(self, individual, verbose=False):
        y_prime = f_model(transform_individual(np.array(individual), self.scaler), self.model)
        d = self.distance(individual)
        s = self.sparsity(individual)
        immutable_score, ranges_score = self.violation(individual, verbose=verbose)
        y_score = 0
        if y_prime == self.original_prediction:
            y_score -= 10000
        else:
            y_score += 10000
        if self.constraints != {} or self.immutables != []:
            immutable_score, ranges_score = self.violation(individual, verbose=verbose)
        else:
            immutable_score, ranges_score = 0, 0
        if verbose:
            print(f"Distance: {d}, Sparsity: {s}, Immutable_score: {immutable_score}, Ranges_score: {ranges_score}")
        return - self.lambda1 * d - self.lambda2 * s + self.lambda3 * y_score + self.lambda4 * immutable_score + self.lambda5 * ranges_score
    
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
                genes.append(self.inverse_transformed_x_features[feature_name])
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
                    genes.extend([self.inverse_transformed_x_features[f] for f in one_hot_group])
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
                    genes.append(self.inverse_transformed_x_features[feature_name])
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
        skip_mutation_indexes = []
        for i in range(len(self.feature_columns)):
            if i in skip_mutation_indexes:
                continue
            if not self.complete_random:
                self.seed_update_number += 1
                self.set_seed(self.seed_number + self.seed_update_number)
                # print(f"Mutation per attribute Seed number: {self.seed_number + self.seed_update_number}")
            if random.random() < self.mutpb:
                feature_name = self.feature_columns[i]
                if i in self.immutables:
                    if feature_name in self.one_hot_encode_features:
                        one_hot_group = [f for f in self.one_hot_encode_features if f.startswith(feature_name.split('_')[0])]
                        skip_mutation_indexes += [list(self.feature_columns).index(f) for f in one_hot_group]
                        for index in skip_mutation_indexes:
                            individual.genes[index] = self.inverse_transformed_x_indexes[index]
                    elif self.inverse_transformed_x_indexes[i] == individual.genes[i]:
                        continue
                    else:
                        individual.genes[i] = self.inverse_transformed_x_indexes[i]
                        continue
                elif feature_name in self.categorical_columns:
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
    
    def fitness_assignment(self, population, clear_fitness=False, verbose=False):
        """
        Assign fitness values to the population based on the evaluation function.
        
        Args:
            population (list): The population of individuals to evaluate.
        """
        if not self.complete_random:
            self.set_seed(self.seed_number)
        if clear_fitness:
            for ind in population: # recalculate the fitness for the entire population
                ind.fitness = None
                ind.fitness = self.evaluate(ind.genes, verbose=verbose)
        else:
            for ind in population:
                ind.fitness = ind.fitness if ind.fitness is not None else self.evaluate(ind.genes, verbose=verbose)
            
    def population(self, population_size=100):
        population = []
        unique_individuals = set()
        while len(population) < population_size:
            new_individual = self.generate_individual()
            genes_tuple = tuple(new_individual.genes)
            if genes_tuple not in unique_individuals:
                population.append(new_individual)
                unique_individuals.add(genes_tuple)
        return population

    def evolve(self):
        # print("Initial seed number: ", self.seed_number + self.seed_update_number)
        
        def generations(population, num_generations, best_fitness):
            # print("Starting evolution...")
            # Initialize variables to track improvements
            generations_without_improvement = 0
            elite_count = max(1, int(self.elite_ratio * len(population)))  # Calculate the number of elite individuals

            for gen in range(num_generations):
                # Termination criterion based on lack of improvement
                if generations_without_improvement >= self.early_stopping_iterations:
                    print("Stopping early due to lack of fitness improvement.")
                    break
                if not self.complete_random:
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
                    if not self.complete_random:
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
                    if not self.complete_random:
                        self.seed_update_number += 1  
                        self.set_seed(self.seed_number + self.seed_update_number)
                        # print(f"Mutation Seed number: {self.seed_number + self.seed_update_number}")

                    if random.random() < self.mutpb:
                        self.mutate_individual(mutant)
                
                # Evaluate the fitness of the offspring
                self.fitness_assignment(offspring)
                # print(f"    Population size: {len(population)}, offspring size: {len(offspring)}")

                # Ensure only the needed number of offspring are retained
                if self.elite_ratio > 0:
                    ## keep the best individuals from the offspring (offspring_size)
                    offspring = sorted(offspring, key=lambda ind: ind.fitness, reverse=True)[:offspring_size]
                    
                    # Combine elite individuals with the new offspring to form the next generation
                    population = elite_individuals + offspring
                else:
                    ## keep the best individuals from the current population and the offspring
                    population = sorted(population + offspring, key=lambda ind: ind.fitness, reverse=True)[:self.population_size]

                # print(" Final population size: ", len(population))
                current_best_fitness, avg_fitness = self.max_avg_fitness(population)
                print(f"    Average fitness: {avg_fitness}, max fitness: {current_best_fitness}")
                # Track improvement
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1                    
            # print("Evolution complete.\n")
            return population
        
        regeneration_tries = 0
        while 1:
            population = self.population(self.population_size)
            print(f"    Diversity: {100 - self.identical_individuals_percentage(population):.2f}% unique individuals")
            
            self.fitness_assignment(population)
            print(f"Constraints: {self.constraints}")
            print(f"Immutable features: {self.immutables}")
                
            max_fitness, avg_fitness = self.max_avg_fitness(population)
            print(f"Initial population average fitness: {avg_fitness}, max fitness: {max_fitness}")
            best_fitness = float("-inf")
            population = generations(population, self.num_generations, best_fitness)
            regeneration_tries += 1
            best_individuals = self.best_individuals(population, self.diversity_top_k)
            
            print(f"Best cfe is: {best_individuals.genes}, guaranteed to alter the decision of the model from {self.original_prediction} to: {f_model(transform_individual(np.array(best_individuals.genes), self.scaler), self.model)}")

            if self.original_prediction == f_model(transform_individual(np.array(best_individuals.genes), self.scaler), self.model)\
            and regeneration_tries < self.regeneration_tries:
                print("No solution found, starting all over again with different initial population.")
                if not self.complete_random:
                    self.seed_number += 1
                continue
            else:
                break   
            
        changed_count, changed_percentage = self.changed_prediction_percentage(population)
        print(f"    Applicable cfes: {changed_count} ({changed_percentage:.2f}%)")
        print(f"    Diversity: {100 - self.identical_individuals_percentage(population):.2f}% unique individuals")
        cfe_with_feature_names = dict(zip(self.feature_columns, best_individuals.genes))  # Transform the best individual list to a dictionary format
        display_cfe_comparison(self.inverse_transformed_x_features, cfe_with_feature_names)
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
                regeneration_tries = 0
                
                while 1:
                    if not self.complete_random:
                        self.seed_number += 1
                        print("SEED NUMBER:  ",self.seed_number)
                    print(f"Constraints: {self.constraints}")
                    print(f"Immutable features: {self.immutables}")
                    # Update the fitness values based on the new constraints
                    self.fitness_assignment(population, clear_fitness=True)
                    
                    max_fitness = max(population, key=lambda ind: ind.fitness).fitness
                    print(f"    !! Best fitness using the new constraints: {max_fitness}")
                    if self.fix_population:
                        population = self.update_population(population)
                        self.fitness_assignment(population, clear_fitness=True)
                        max_fitness = max(population, key=lambda ind: ind.fitness).fitness
                        print(f"    !! Best fitness after fixing the population: {max_fitness}")
                       
                    if self.population_size_dynamic > 0:
                        ## Just create 1 individual that adheres to the new constraints
                        new_population = self.population(population_size=self.population_size_dynamic)

                        ## get the fitness of the new individual
                        self.fitness_assignment(new_population)
                        ## print the fitness of the best new population
                        print(f"    !! Best fitness of the brand new population: {max(new_population, key=lambda ind: ind.fitness).fitness}")

                        ## now add this individual to the population 
                        population.extend(new_population)
                                        
                        # Update the fitness values based on the new constraints
                        self.fitness_assignment(population, clear_fitness=True)
                        print(f"    !! Best fitness of the updated population: {max(new_population, key=lambda ind: ind.fitness).fitness}")

                        # and remove the worst individual cause we just added a new one
                        population = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:-self.population_size_dynamic]
                        
                        # count the population size
                        print(f"Population size: {len(population)}")
                    
                    print(f"Constraints: {self.constraints}")
                    print(f"Immutable features: {self.immutables}")
                    start_time = time()
                    # self.cxpb = 0.9
                    # self.mutpb = 0.8
                    # self.elite_ratio = 0.1
                    population = generations(population, 30, max_fitness)
                    best_individuals = self.best_individuals(population, self.diversity_top_k)
                    print(f"Time taken for dynamic constraint placement: {time() - start_time:.2f} seconds")
                    print(f"Best cfe is: {best_individuals.genes}, guaranteed to alter the decision of the model from {self.original_prediction} to: {f_model(transform_individual(np.array(best_individuals.genes), self.scaler), self.model)}")

                    if self.original_prediction == f_model(transform_individual(np.array(best_individuals.genes), self.scaler), self.model)\
                                    and regeneration_tries < self.regeneration_tries:
                        print("No solution found, starting all over again with different initial population.")
                        if not self.complete_random:
                            self.seed_number += 1
                            self.set_seed(self.seed_number)
                        regeneration_tries += 1
                        continue
                    else:
                        break     
                
                changed_count, changed_percentage = self.changed_prediction_percentage(population)
                print(f"    Applicable cfes: {changed_count} ({changed_percentage:.2f}%)")
                print(f"    Diversity: {100 - self.identical_individuals_percentage(population):.2f}% unique individuals")
                cfe_with_feature_names = dict(zip(self.feature_columns, best_individuals.genes))  # Transform the best individual list to a dictionary format
                display_cfe_comparison(self.inverse_transformed_x_features, cfe_with_feature_names)
                print()
                accepted = self.ask_user_acceptance()
            return best_individuals
        else:
            return best_individuals
        
    def update_population(self, population):
        ## deepcopy the population
        population = deepcopy(population)
        for ind in population:
            skip_indices = []
            
            for i in range(len(self.feature_columns)):
                if i in skip_indices:
                    continue
                feature_name = self.feature_columns[i]
                
                if i in self.immutables:    
                    if feature_name in self.one_hot_encode_features:
                        one_hot_group = [f for f in self.one_hot_encode_features if f.startswith(feature_name.split('_')[0])]
                        skip_indices.extend([list(self.feature_columns).index(f) for f in one_hot_group])
                        for index in skip_indices:
                            ind.genes[index] = self.inverse_transformed_x_indexes[index]
                    elif self.inverse_transformed_x_indexes[i] == ind.genes[i]:
                        continue
                    else:
                        ind.genes[i] = self.inverse_transformed_x_indexes[i]
                        continue
                elif feature_name in self.categorical_columns:
                    continue
                elif feature_name in self.one_hot_encode_features:
                    continue
                else:
                    if self.constraints.get(i):
                        if ind.genes[i] == self.inverse_transformed_x_indexes[i]:
                            continue
                        else:
                            lower, upper = self.constraints[i]
                            if lower < ind.genes[i] < upper:
                                continue
                            else:
                                if self.features_type[feature_name] == 'int':
                                    ind.genes[i] = random.randint(lower, upper)
                                else:
                                    ind.genes[i] = random.uniform(lower, upper)
                    else:
                        continue      
        return population
     
    def changed_prediction_individuals(self, population):
        """
        Find the unique applicable CFEs, count them, and calculate the percentage compared to the unique individuals in a population.
        """
        # Use a dictionary to keep track of unique individuals based on their genes as the key
        unique_individuals = {tuple(ind.genes): ind for ind in population}

        # Find the unique individuals that change the prediction
        unique_applicable_cfes = [
            ind for genes, ind in unique_individuals.items()
            if f_model(transform_individual(np.array(genes), self.scaler), self.model) != self.original_prediction
        ]
        
        identical_individuals_percentage = (len(unique_individuals) / len(population)) * 100

        unique_applicable_cfes_len = len(unique_applicable_cfes)
        unique_applicable_cfes_to_unique_individuals_percentage = (unique_applicable_cfes_len / len(unique_individuals)) * 100

        return unique_applicable_cfes, identical_individuals_percentage, unique_applicable_cfes_len, unique_applicable_cfes_to_unique_individuals_percentage

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
    def ask_user_acceptance(self, best_individual=None):   
        # Mock user input (replace with actual user input handling)
        while True:
            if not self.automatic_user_acceptance:
                user_response = input("Do you accept this counterfactual explanation? (y/n): ").strip().lower()
            else:
                user_response = self.check_constraints(best_individual)
            if user_response in ["y", "n"]:
                return user_response == "y"
            else:
                print("Invalid input. Please type 'y' or 'n'.")

    def check_constraints(self, best_individual):
        skip_indexes = []
        for i in range(len(self.feature_columns)):
            if i in skip_indexes:
                continue
            feature_name = self.feature_columns[i]

            if i in self.immutables:
                if feature_name in self.one_hot_encode_features:
                    one_hot_group = [f for f in self.one_hot_encode_features if f.startswith(feature_name.split('_')[0])]
                    skip_indexes += [list(self.feature_columns).index(f) for f in one_hot_group]
                    for index in skip_indexes:
                        if best_individual.genes[index] != self.inverse_transformed_x_indexes[index]:
                            return False
                elif self.inverse_transformed_x_indexes[i] != best_individual.genes[i]:
                    return False
                
            elif feature_name in self.categorical_columns and self.data_distribution:
                if best_individual.genes[i] not in self.features_ranges[feature_name]:
                    return False
                
            elif feature_name in self.one_hot_encode_features:
                one_hot_group = [f for f in self.one_hot_encode_features if f.startswith(feature_name.split('_')[0])]
                skip_indexes += [list(self.feature_columns).index(f) for f in one_hot_group]
                for index in skip_indexes:
                    if best_individual.genes[index] != self.inverse_transformed_x_indexes[index]:
                        return False
                    
            else:
                if self.constraints.get(i):
                    lower, upper = self.constraints[i]
                    if not lower <= best_individual.genes[i] <= upper:
                        return False
        return True

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
                    if feature in self.updated_constraints:
                        user_input = self.updated_constraints[feature]
                    else:
                        user_input = input(f"Set '{base_feature_name}' as immutable? ('i' for immutable, 'ni' to remove immutability or Enter to skip): ").strip()
                    
                    find_indices = [i for i, f in enumerate(self.feature_columns) if f.startswith(base_feature_name)]
                    skip_indices.extend(find_indices)
                    
                    if user_input == "":
                        print(f"Feature '{base_feature_name}' is passed (no new constraints).")
                        break
                    elif user_input == "i":
                        for idx in find_indices:
                            if idx not in self.immutables:
                                self.immutables.append(idx)
                        print(f"Feature '{base_feature_name}' marked as immutable.")
                        break
                    elif user_input == "ni":
                        ## remove the immutable constraint
                        ## pop the skip indices from the self.immutables
                        for idx in find_indices:
                            self.immutables.remove(idx)                        
                        print(f"Feature '{base_feature_name}' is no longer immutable.")
                        break
            else:
                while True:
                    if feature in self.updated_constraints:
                        user_input = self.updated_constraints[feature]
                    else:
                        user_input = input(f"Enter the lower and upper bounds for feature '{feature}' (or 'i' to make it immutable or 'ni' to remove immutability, Enter to skip): ").strip()
                    
                    # User presses enter to pass the feature (no change)
                    if user_input == "":
                        print(f"Feature '{feature}' is passed (no new constraints).")
                        break
                    
                    # User enters '-' to mark the feature as immutable
                    elif user_input == "i":
                        self.immutables.append(i)
                        ## if the feature is in the constraints then remove it
                        if i in self.constraints:
                            self.constraints.pop(i)
                            print(f"Feature '{feature}' marked as immutable. Constraints removed.")
                        else:
                            print(f"Feature '{feature}' marked as immutable.")                        
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
                                ## if the feature is in the immutable list then remove it
                                if i in self.immutables:
                                    self.immutables.remove(i)
                                    print(f"Feature '{feature}' set to [{lower}, {upper}] and is no longer immutable.")
                                else:
                                    print(f"Feature '{feature}' set to [{lower}, {upper}].")
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