from scipy.spatial.distance import pdist
from sklearn.neighbors import KDTree
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
import random
import heapq
import copy
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

from utils import f_model, display_cfe_comparison,\
    required_attributes, initial_label_encode_data, label_encode_data,\
    decode_label_encoded_data, normalize_data, safe_divide,\
    compute_distances_in_blocks, fast_max_l2_l1_distance

from feature_weights import get_feature_weights

class Individual:
    def __init__(self, genes, fitness=None):
        self.genes = np.array(genes, dtype=np.float64)
        self.fitness = fitness
        self.normalized = None
        self.decoded = None
              
    def set_fitness(self, fitness):
        self.fitness = fitness
        
    def copy(self):
        """Creates a deep copy of the Individual instance."""
        new_individual = Individual(self.genes.copy(), self.fitness)
        new_individual.normalized = copy.deepcopy(self.normalized)  
        new_individual.decoded = copy.deepcopy(self.decoded)
        return new_individual
        
    def __repr__(self):
        return f"Individual(genes={self.genes}, fitness={self.fitness}, normalized={self.normalized}, decoded={self.decoded})"

class UGCE:
    def __init__(self, model, dataset,\
        numerical_columns, categorical_columns):
        self.model = model
        self.dataset = dataset
        self.dataset_length = len(self.dataset)
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.feature_names = self.dataset.columns
        self.feature_columns_len = len(self.feature_names)
        self.positive_mask = self.model.predict(dataset) == 1
        self.positive_indices = dataset.index[self.positive_mask]
        self.one_hot_encode_features = set(self.feature_names).difference(set(self.numerical_columns))
        self.dataset, self.categorical_label_encoders = initial_label_encode_data(dataset, self.feature_names, self.categorical_columns)
        self.min_max_scaler_per_column, self.features_ranges, self.features_type=required_attributes(self.dataset)
        self.normalized_dataset = normalize_data(self.dataset, self.feature_names, self.dataset)
        if self.dataset_length >= 60000:
            self.max_l2_distance, self.max_l1_distance = fast_max_l2_l1_distance(self.normalized_dataset.values)
        else:
            # self.max_l2_distance = np.max(compute_distances_in_blocks(self.normalized_dataset.values, block_size=100, representation=16))
            self.max_l2_distance = np.max(pdist(self.normalized_dataset.values))
            self.max_l1_distance = np.max(pdist(self.normalized_dataset.values, metric='cityblock'))
        self.max_sparsity = len(self.feature_names)
        self.feature_weights_list = get_feature_weights(self.dataset, self.feature_names, self.numerical_columns, self.features_ranges, feature_weights="inverse_mad", encoding="label")   
        self.numerical_feature_indexes = [i for i, feature_name in enumerate(self.feature_names) if feature_name in numerical_columns]
  
    def set_seed(self, seed_number):
        """Set seeds for reproducibility in random and np.random."""
        random.seed(seed_number)
        np.random.seed(seed_number)

    def tournament_selection(self, population, k):
        """
        Perform tournament selection from a population.

        This method selects `k` individuals from the population by holding
        `self.tournsize` tournaments. In each tournament, `self.tournsize` individuals
        are randomly chosen, and the one with the highest fitness is selected.

        Parameters:
        - population: List of individuals from which to select.
        - k: Number of individuals to select.

        Returns:
        - List of selected individuals.
        """
        winners = []
        for _ in range(k):
            participants = random.sample(population, self.tournsize)
            winners.append(max(participants, key=lambda ind: ind.fitness))
        return winners

    def roulette_selection(self, population, k):
        """
        Perform roulette wheel selection from a population.

        This method selects `k` individuals based on their fitness proportionally. 
        Each individual's chance of being selected is proportional to its fitness,
        simulating a roulette wheel spin.

        Parameters:
        - population: List of individuals from which to select.
        - k: Number of individuals to select.

        Returns:
        - List of selected individuals.
        """
        selected_roulette = []
        min_fitness = min(ind.fitness for ind in population)
        offset = abs(min_fitness) if min_fitness < 0 else 0
        adjusted_population = [ind.fitness + offset for ind in population]
        total_adjusted_fitness = sum(adjusted_population)

        for _ in range(k):
            pick = random.uniform(0, total_adjusted_fitness)
            current = 0
            for ind, adjusted_fitness in zip(population, adjusted_population):
                current += adjusted_fitness
                if current > pick:
                    selected_roulette.append(ind)
                    break
        return selected_roulette

    def rank_selection(self, population, k):
        """
        Perform rank-based selection from a population.

        This method sorts the population by fitness and assigns a selection
        probability inversely proportional to the rank. Individuals are selected
        using these probabilities, allowing for a fair chance across different
        fitness levels.

        Parameters:
        - population: List of individuals from which to select.
        - k: Number of individuals to select.

        Returns:
        - List of selected individuals.
        """
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        
        total_ranks = len(population) * (len(population) + 1) / 2
        rank_probabilities = [(len(population) - i) / total_ranks for i in range(len(population))]
        selected = random.choices(sorted_population, weights=rank_probabilities, k=k)
        return selected

    def sus_selection(self, population, k):
        """
        Perform stochastic universal sampling (SUS) from a population.

        SUS is a method that selects `k` individuals by using a single random start
        point and evenly spaced intervals, ensuring a spread of selections across
        the population's fitness range.

        Parameters:
        - population: List of individuals from which to select.
        - k: Number of individuals to select.

        Returns:
        - List of selected individuals.
        """
        selected_sus = []
        min_fitness = min(ind.fitness for ind in population)
        offset = abs(min_fitness) if min_fitness < 0 else 0
        adjusted_population = [ind.fitness + offset for ind in population]

        total_adjusted_fitness = sum(adjusted_population)
        distance = total_adjusted_fitness / k
        start = random.uniform(0, distance)
        pointers = [start + i * distance for i in range(k)]

        for pointer in pointers:
            cumulative = 0
            for ind, adjusted_fitness in zip(population, adjusted_population):
                cumulative += adjusted_fitness
                if cumulative >= pointer:
                    selected_sus.append(ind)
                    break
        return selected_sus

    def fifty_percent_best_selection(self, population, k):
        """
        Perform 50% best selection from a population.

        This method selects the top 50% of individuals based on their fitness.

        Parameters:
        - population: List of individuals from which to select.
        - k: Number of individuals to select.

        Returns:
        - List of selected individuals.
        """
        return sorted(population, key=lambda ind: ind.fitness, reverse=True)[:k]
    
    def update_individuals(self, cfes):
        """
        Update the normalized and decoded gsenes of the individuals.
        
        Parameters:
        - cfes: List of individuals to update.

        Returns:
        - None.
        """
        all_genes = np.array([ind.genes for ind in cfes])
        normalized_genes = normalize_data(all_genes, self.feature_names, self.dataset)
        decoded_genes_df = pd.DataFrame(decode_label_encoded_data(all_genes, self.feature_names,
                                    self.categorical_columns, self.categorical_label_encoders),
                                    columns=self.feature_names)
        for ind, norm, (_, decoded) in zip(cfes, normalized_genes, decoded_genes_df.iterrows()):
            ind.normalized = norm
            ind.decoded = decoded.values
   
    def explain_instances(self, instances_to_explain, constraints=None, immutables=None,\
        evaluation=False, dynamic_constraints=False,\
        initial_population_variability=0.2, data_distribution=True,\
        seed_number=42, strategy="fix_population_update_fitness",\
        initial_population_strategy = "random", population_size_dynamic=0,\
        num_generations=50, population_size=50, regeneration_tries_intermediate=5,\
        regeneration_tries_intermediate_after_updating_constraints= 1, num_parents=10,\
        cfes_requested=1, selection_method="tournament", tournsize=3,\
        early_stopping_iterations=3, thresh=1e-2, elite_ratio=0.1, \
        cxpb=0.5, crossover_points=3, mutpb=0.2, distance_metric="weighted_l1",\
        updated_constraints=None, multistep_updated_constraints=False, automatic_user_acceptance=True, verbose=False,
        running_times_per_instance=10, normalized_fitness=True, reweighting_lambdas_after_generations=-1,
        initial_lambdas_without_constraints={"lambda1":0.2, "lambda2":0.2, "lambda3":1},
        initial_lambdas_with_constraints={"lambda1":0.2, "lambda2":0.2, "lambda3":1},
        lambdas_after_updating_constraints={"lambda1":0.2, "lambda2":0.2, "lambda3":1},
        lambdas_reweighting_after_updating_constraints_and_some_generations={"lambda1":0.2, "lambda2":0.2, "lambda3":1}):
        """
        Generates counterfactual explanations by evolving a population of candidate counterfactuals.

        ### Parameters:
        - **instances_to_explain** (*pd.DataFrame*): The dataset instances to explain.
        - **constraints** (*dict, default=None*): Dictionary specifying constraints on feature changes.
        - **immutables** (*list, default=None*): List of features that must remain unchanged.
        - **evaluation** (*bool, default=False*): Whether to evaluate counterfactual quality using predefined metrics.
        - **dynamic_constraints** (*bool, default=False*): Whether to apply dynamically updated constraints during evolution.
        - **initial_population_variability** (*float, default=0.2*): Controls diversity in the initial population (higher means more variation).
        - **data_distribution** (*bool, default=True*): Whether to sample new instances following the original data distribution.
        - **seed_number** (*int, default=42*): Random seed for reproducibility.
        - **strategy** (*str, default="fix_population_update_fitness"*): Defines how the population is evolved.
        - **initial_population_strategy** (*str, default="random"*): Strategy for initializing the population (`"random"` or `"guided"`).
        - **population_size_dynamic** (*int, default=0*): If >0, allows population size to change dynamically.
        - **num_generations** (*int, default=50*): Number of generations to evolve the population.
        - **population_size** (*int, default=50*): Number of individuals in each generation.
        - **regeneration_tries_intermediate** (*int, default=5*): Maximum attempts to regenerate a valid population before giving up.
        - **regeneration_tries_intermediate_after_updating_constraints** (*int, default=1*): Regeneration attempts after constraints are updated.
        - **num_parents** (*int, default=10*): Number of parents selected per generation for reproduction.
        - **cfes_requested** (*int, default=1*): Number of counterfactuals to generate per instance.
        - **selection_method** (*str, default="tournament"*): Selection strategy (`"tournament"`, `"roulette"`, `"rank"`, `"sus"`, `"50percentbest"`).
        - **tournsize** (*int, default=3*): Size of the tournament selection group.
        - **early_stopping_iterations** (*int, default=3*): Stops evolution if no improvement is observed for this many generations.
        - **thresh** (*float, default=1e-2*): Threshold for determining if a generation has improved.
        - **elite_ratio** (*float, default=0.1*): Proportion of top individuals to retain between generations.
        - **cxpb** (*float, default=0.5*): Probability of applying crossover to selected parents.
        - **crossover_points** (*int, default=3*): Number of crossover points during mating.
        - **mutpb** (*float, default=0.2*): Probability of mutating an offspring.
        - **updated_constraints** (*dict, default=None*): Constraints to apply after the evolution process starts.
        - **automatic_user_acceptance** (*bool, default=True*): If `True`, automatically accepts counterfactuals that meet constraints.
        - **verbose** (*bool, default=False*): Whether to print detailed logs during execution.
        - **running_times_per_instance** (*int, default=10*): Number of times to run the pipeline for each instance.
        - **normalized_fitness** (*bool, default=True*): Whether to normalize fitness scores.
        - **reweighting_lambdas_after_generations** (*int, default=-1*): Number of generations before adjusting the lambda values.
        
        ### **Lambda Weight Parameters** (For loss function balancing):
        - **initial_lambdas_without_constraints** (*dict*): Weights used when constraints are NOT applied.
        - `"lambda1": 0.25`, `"lambda2": 0.25`, `"lambda3": 0.5`, `"lambda4": 0.0`, `"lambda5": 0.0`
        - **initial_lambdas_with_constraints** (*dict*): Weights used when constraints ARE applied.
        - `"lambda1": 0.1`, `"lambda2": 0.1`, `"lambda3": 0.3`, `"lambda4": 0.25`, `"lambda5": 0.25`
        - **lambdas_after_updating_constraints** (*dict*): Weights used after constraints are dynamically updated.
        - `"lambda1": 0.1`, `"lambda2": 0.1`, `"lambda3": 0.3`, `"lambda4": 0.25`, `"lambda5": 0.25`
        - **lambdas_reweighting_after_updating_constraints_and_some_generations** (*dict*): Weights used after multiple updates.
        - `"lambda1": 0.25`, `"lambda2": 0.25`, `"lambda3": 0.3`, `"lambda4": 0.1`, `"lambda5": 0.1`

        ### **Returns:**
        - **List of best individuals for each instance** (*list*): Counterfactual explanations found during evolution.
        """
        self.initial_population_strategy = initial_population_strategy
        self.seed_number = seed_number
        self.set_seed(self.seed_number)

        prepare_instances_time = time()
        self.instances_to_explain = label_encode_data(instances_to_explain, self.feature_names, self.categorical_columns,\
                                                        self.categorical_label_encoders)
        self.instances_to_explain_numpy = instances_to_explain.to_numpy()
        self.instances_to_explain_normalized = normalize_data(self.instances_to_explain, self.feature_names, self.dataset)
        prepare_instances_time = time() - prepare_instances_time
        
        population_generation_time_global = 0
        if self.initial_population_strategy == "kdtree":
            population_generation_time_global = time()
            # Build KDTree for the dataset
            ## KDTree requires data to be numerical, so provide the fully transformed dataset.
            #  and query with the fully transformed instance
            ## KDTree should only be fitted and queried with the numerical columns
            ## if the datasat is very large, sample the dataset to build the KDTree
            self.kdTree_dataset = None
            if self.dataset_length >= 60000:
                self.kdTree_dataset = self.dataset.sample(n=60000, random_state=self.seed_number)
            else:
                self.kdTree_dataset = self.dataset
            self.kdTree = KDTree(self.kdTree_dataset)
            population_generation_time_global = time() - population_generation_time_global

        self.evaluation = evaluation
        self.dynamic_constraints = dynamic_constraints
        self.initial_population_variability = initial_population_variability
        self.num_generations = num_generations
        if early_stopping_iterations == -1:
            self.early_stopping_iterations = num_generations
        else:
            self.early_stopping_iterations = early_stopping_iterations
        if thresh == -1:
            self.thresh = 1e-2
        else:
            self.thresh = thresh
        self.elite_ratio = elite_ratio
        self.data_distribution = data_distribution

        self.strategy = strategy
        self.population_size_dynamic = population_size_dynamic
        self.num_generations = num_generations
        self.population_size = population_size * cfes_requested
        self.regeneration_tries_intermediate = regeneration_tries_intermediate
        self.regeneration_tries_intermediate_after_updating_constraints = regeneration_tries_intermediate_after_updating_constraints
        self.num_parents = num_parents
        self.selection_method = selection_method
        self.tournsize = tournsize

        self.cxpb = cxpb
        self.crossover_points = crossover_points
        self.mutpb = mutpb
        self.distance_metric = distance_metric
        self.updated_constraints = updated_constraints
        self.multistep_updated_constraints = multistep_updated_constraints
        self.automatic_user_acceptance = automatic_user_acceptance
        self.verbose = verbose
        self.normalized_fitness = normalized_fitness
        self.constraints = constraints if constraints else {}
        if self.constraints == {} or self.constraints is None:
            self.lambda1 = initial_lambdas_without_constraints["lambda1"]
            self.lambda2 = initial_lambdas_without_constraints["lambda2"]
            self.lambda3 = initial_lambdas_without_constraints["lambda3"]
        else:
            self.lambda1 = initial_lambdas_with_constraints["lambda1"]
            self.lambda2 = initial_lambdas_with_constraints["lambda2"]
            self.lambda3 = initial_lambdas_with_constraints["lambda3"]
        self.lambdas_after_updating_constraints = lambdas_after_updating_constraints
        self.lambdas_reweighting_after_updating_constraints_and_some_generations = lambdas_reweighting_after_updating_constraints_and_some_generations
        self.reweighting_lambdas_after_generations = reweighting_lambdas_after_generations
        self.immutables = immutables if immutables else []
        self.setup_constraints()
               
        results_X = {}
        iteration = 0

        empty_intermediate_counter = 0

        for i, x in tqdm(instances_to_explain.iterrows(), total=instances_to_explain.shape[0]):
            num_generations = 0

            time_intermediate_from_scratch = time()
            self.x = x
            self.x_dict = self.x.to_dict()
            self.x_numpy = self.x.to_numpy()
            self.x_label_encoded = self.instances_to_explain.loc[i].to_frame().T
            self.x_label_encoded_numpy = self.x_label_encoded.to_numpy()[0]
            self.normalized_x = self.instances_to_explain_normalized.loc[i].to_frame().T
            self.normalized_x_numpy = self.normalized_x.to_numpy()[0]
            
            time_intermediate_from_scratch = time() - time_intermediate_from_scratch
            results = {
                "Single_run": {
                    "Original_Instance": self.x_numpy,
                    "Encoded_Instance": self.x_label_encoded_numpy,
                    "Normalized_Instance": self.normalized_x_numpy,
                    "Best_cfe": np.array([]),
                    "Encoded_Best_cfe": np.array([]),
                    "Normalized_Best_cfe": np.array([]),
                    "Best_cfe_fitness": -np.inf,
                    "Best_cfe_distance": np.inf,
                    "Best_cfe_l2_distance": np.inf,
                    "Best_cfe_l1_distance": np.inf,
                    "Best_cfe_weighted_l1_distance": np.inf,
                    "Best_cfe_sparsity": None,

                    "Intermediate_best_cfe": [],
                    "Encoded_Intermediate_best_cfe": [],
                    "Normalized_Intermediate_best_cfe": [],
                    "Intermediate_best_cfe_fitness": -np.inf,
                    "Intermediate_best_cfe_distance": np.inf,
                    "Intermediate_best_cfe_l2_distance": np.inf,
                    "Intermediate_best_cfe_l1_distance": np.inf,
                    "Intermediate_best_cfe_weighted_l1_distance": np.inf,
                    "Intermediate_best_cfe_sparsity": None,
                    
                    "Distance_between_best_and_intermediate_best_cfes": np.inf,
                    
                    "Applicable_cfes_number": None,
                    "Intermediate_applicable_cfes_number": None,

                    "Time_intermediate_from_scratch": None,
                    "Time_dynamic": None,

                    "Cfes": [],
                    "Num_generations": None,
                },
                "Multiple_runs": {
                    "Avg_Best_cfe_distance": 0,
                    "Avg_Best_cfe_l2_distance": 0,
                    "Avg_Best_cfe_l1_distance": 0,
                    "Avg_Best_cfe_weighted_l1_distance": 0,
                    "Avg_Best_cfe_sparsity": 0,

                    "Avg_Intermediate_best_cfe_distance": 0,
                    "Avg_Intermediate_best_cfe_l2_distance": 0,
                    "Avg_Intermediate_best_cfe_l1_distance": 0,
                    "Avg_Intermediate_best_cfe_weighted_l1_distance": 0,
                    "Avg_Intermediate_best_cfe_sparsity": 0,

                    "Avg_applicable_cfes_number": 0,
                    "Avg_intermediate_applicable_cfes_number": 0,
                    "Avg_number_of_generations": 0,

                    "Avg_distance_between_best_and_intermediate_best_cfes": 0,
                    "Avg_l2_distance_between_best_and_intermediate_best_cfes": 0,
                    "Avg_l1_distance_between_best_and_intermediate_best_cfes": 0,
                    "Avg_weighted_l1_distance_between_best_and_intermediate_best_cfes": 0,

                    "Times_that_at_least_one_cfe_found_percentage": 0,
                    "Times_that_at_least_one_intermediate_cfe_found_percentage": 0,
                    
                    "Avg_Time_intermediate_from_scratch": 0,
                    "Avg_Time_dynamic": 0
                }
            }
            times_intermediate_and_final_cfe_found = 0
            population_generation_time_per_run = []

            for _ in range(running_times_per_instance):
                time_intermediate_from_scratch_per_run = time()
                self.original_prediction = f_model(self.x.to_frame().T, self.model)
                time_intermediate_from_scratch_per_run = time() - time_intermediate_from_scratch_per_run

                unique_applicable_cfes_intermediate, cfes, population_generation_time, elapsed_time_intermediate_from_scratch,\
                      time_intermediate_dynamic, num_generations, _ = self.evolve()
                time_intermediate_from_scratch_per_run = time_intermediate_from_scratch_per_run + elapsed_time_intermediate_from_scratch + time_intermediate_from_scratch
                population_generation_time_per_run.append(population_generation_time)
                len_cfes = len(cfes)
                if len_cfes > 0:
                    # Update individuals with normalized and decoded genes
                    self.update_individuals(cfes)
                    best_individual = None
                    #############################################################
                    ############## STATS ABOUT THE FINAL SOLUTION ###############
                    #############################################################
                    if len_cfes == 1:
                        best_individual = cfes[0]
                        results["Single_run"]['Best_cfe'] = best_individual.decoded
                        results["Single_run"]['Encoded_Best_cfe'] = best_individual.genes
                        results["Single_run"]['Best_cfe_fitness'] = best_individual.fitness
                        normalized_individual = best_individual.normalized
                        results["Single_run"]["Normalized_Best_cfe"] = normalized_individual

                        l2_dist = self.normalized_l2_distance(normalized_individual)
                        results["Single_run"]['Best_cfe_l2_distance'] = l2_dist
                        l1_dist = self.normalized_l1_distance(normalized_individual)
                        results["Single_run"]['Best_cfe_l1_distance'] = l1_dist
                        weighted_l1_dist = self.compute_proximity_loss_dice(normalized_individual)
                        results["Single_run"]['Best_cfe_weighted_l1_distance'] = weighted_l1_dist

                        if self.distance_metric == "l2":
                            results["Single_run"]['Best_cfe_distance'] = l2_dist
                            results["Multiple_runs"]['Avg_Best_cfe_distance'] += l2_dist
                        elif self.distance_metric == "l1":
                            results["Single_run"]['Best_cfe_distance'] = l1_dist
                            results["Multiple_runs"]['Avg_Best_cfe_distance'] += l1_dist
                        elif self.distance_metric == "weighted_l1":
                            results["Single_run"]['Best_cfe_distance'] = weighted_l1_dist
                            results["Multiple_runs"]['Avg_Best_cfe_distance'] += weighted_l1_dist

                        sparsity = self.normalized_sparsity(normalized_individual)
                        results["Single_run"]["Best_cfe_sparsity"] = sparsity
                        
                        
                        results["Multiple_runs"]['Avg_Best_cfe_sparsity'] += \
                            results["Single_run"]["Best_cfe_sparsity"]
                    else:
                        best_ind = None
                        best_cfe_distance = np.inf

                        for ind in cfes:
                            dist = np.inf
                            normalized_individual = ind.normalized
                            if self.distance_metric == "l2":
                                dist = self.normalized_l2_distance(normalized_individual)
                            elif self.distance_metric == "l1":
                                dist = self.normalized_l1_distance(normalized_individual)
                            elif self.distance_metric == "weighted_l1":
                                dist = self.compute_proximity_loss_dice(normalized_individual)
                            else:
                                raise ValueError("Error: Distance metric not supported.")
                            if dist < best_cfe_distance:
                                best_cfe_distance = dist
                                best_ind = ind.copy()
                        results["Single_run"]['Best_cfe'] = best_ind.decoded
                        results["Single_run"]['Best_cfe_fitness'] = best_ind.fitness
                        results["Single_run"]["Encoded_Best_cfe"] = best_ind.genes
                        results["Single_run"]["Normalized_Best_cfe"] = best_ind.normalized
                        results["Single_run"]['Best_cfe_distance'] = best_cfe_distance
                        results["Single_run"]["Best_cfe_l2_distance"] = self.normalized_l2_distance(best_ind.normalized)
                        results["Single_run"]["Best_cfe_l1_distance"] = self.normalized_l1_distance(best_ind.normalized)
                        results["Single_run"]["Best_cfe_weighted_l1_distance"] = self.compute_proximity_loss_dice(best_ind.normalized)
                        sparsity = self.normalized_sparsity(best_ind.normalized)
                        results["Single_run"]["Best_cfe_sparsity"] = sparsity

                        results["Multiple_runs"]['Avg_Best_cfe_distance'] += best_cfe_distance
                        results["Multiple_runs"]['Avg_Best_cfe_l2_distance'] += results["Single_run"]["Best_cfe_l2_distance"]
                        results["Multiple_runs"]['Avg_Best_cfe_l1_distance'] += results["Single_run"]["Best_cfe_l1_distance"]
                        results["Multiple_runs"]['Avg_Best_cfe_weighted_l1_distance'] += results["Single_run"]["Best_cfe_weighted_l1_distance"]
                        results["Multiple_runs"]['Avg_Best_cfe_sparsity'] += sparsity

                    results["Single_run"]['Applicable_cfes_number'] = len_cfes
                    results["Single_run"]['Num_generations'] = num_generations

                    results["Multiple_runs"]['Avg_applicable_cfes_number'] += len_cfes
                    results["Multiple_runs"]['Avg_number_of_generations'] += num_generations
                    
                    results["Single_run"]['cfes'] = cfes
                    results["Single_run"]['Time_intermediate_from_scratch'] = time_intermediate_from_scratch_per_run
                    results["Multiple_runs"]['Avg_Time_intermediate_from_scratch'] += time_intermediate_from_scratch_per_run
                    results["Multiple_runs"]['Times_that_at_least_one_cfe_found_percentage'] += 1
                    ###################################################################
                    ################### INTERMEDIATE SOLUTION STATS ###################
                    ###################################################################
                    if self.dynamic_constraints:
                        unique_applicable_cfes_intermediate_len = len(unique_applicable_cfes_intermediate)
                        if unique_applicable_cfes_intermediate_len > 0:
                            # Update individuals with normalized and decoded genes
                            self.update_individuals(unique_applicable_cfes_intermediate)
                            best_ind_intermid = None
                            best_cfe_distance = np.inf

                            for ind in unique_applicable_cfes_intermediate:
                                normalized_individual = ind.normalized
                                dist = np.inf
                                if self.distance_metric == "l2":
                                    dist = self.normalized_l2_distance(normalized_individual)
                                elif self.distance_metric == "l1":
                                    dist = self.normalized_l1_distance(normalized_individual)
                                elif self.distance_metric == "weighted_l1":
                                    dist = self.compute_proximity_loss_dice(normalized_individual)
                                if dist < best_cfe_distance:
                                    best_cfe_distance = dist
                                    best_ind_intermid = ind.copy()

                            results["Single_run"]['Intermediate_best_cfe'] = best_ind_intermid.decoded
                            results["Single_run"]['Intermediate_best_cfe_fitness'] = best_ind_intermid.fitness
                            results["Single_run"]["Scaled_Intermediate_best_cfe"] = best_ind_intermid.genes
                            results["Single_run"]["Normalized_Intermediate_best_cfe"] = best_ind_intermid.normalized
                            
                            results["Single_run"]['Intermediate_best_cfe_distance'] = best_cfe_distance
                            results["Single_run"]["Intermediate_best_cfe_l2_distance"] = self.normalized_l2_distance(best_ind_intermid.normalized)
                            results["Single_run"]["Intermediate_best_cfe_l1_distance"] = self.normalized_l1_distance(best_ind_intermid.normalized)
                            results["Single_run"]["Intermediate_best_cfe_weighted_l1_distance"] = self.compute_proximity_loss_dice(best_ind_intermid.normalized)
                            sparsity = self.normalized_sparsity(best_ind_intermid.normalized)
                            results["Single_run"]["Intermediate_best_cfe_sparsity"] = sparsity

                            results["Multiple_runs"]['Avg_Intermediate_best_cfe_distance'] += best_cfe_distance
                            results["Multiple_runs"]['Avg_Intermediate_best_cfe_l2_distance'] += results["Single_run"]["Intermediate_best_cfe_l2_distance"]
                            results["Multiple_runs"]['Avg_Intermediate_best_cfe_l1_distance'] += results["Single_run"]["Intermediate_best_cfe_l1_distance"]
                            results["Multiple_runs"]['Avg_Intermediate_best_cfe_weighted_l1_distance'] += results["Single_run"]["Intermediate_best_cfe_weighted_l1_distance"]
                            results["Multiple_runs"]['Avg_Intermediate_best_cfe_sparsity'] += sparsity

                            results["Single_run"]['Intermediate_applicable_cfes_number'] = unique_applicable_cfes_intermediate_len
                            results["Multiple_runs"]['Times_that_at_least_one_intermediate_cfe_found_percentage'] += 1
                        
                            
                            times_intermediate_and_final_cfe_found += 1
                            l2_distance_between_best_and_intermediate_best_cfes = self.normalized_l2_distance(\
                                    results["Single_run"]['Normalized_Best_cfe'], results["Single_run"]['Normalized_Intermediate_best_cfe'])
                            l1_distance_between_best_and_intermediate_best_cfes = self.normalized_l1_distance(\
                                    results["Single_run"]['Normalized_Best_cfe'], results["Single_run"]['Normalized_Intermediate_best_cfe'])
                            weighted_l1_distance_between_best_and_intermediate_best_cfes = self.compute_proximity_loss_dice(\
                                    results["Single_run"]['Normalized_Best_cfe'], results["Single_run"]['Normalized_Intermediate_best_cfe'])

                            if self.distance_metric == "l2":
                                results["Single_run"]["Distance_between_best_and_intermediate_best_cfes"] = l2_distance_between_best_and_intermediate_best_cfes
                                results["Multiple_runs"]["Avg_distance_between_best_and_intermediate_best_cfes"] += l2_distance_between_best_and_intermediate_best_cfes
                            elif self.distance_metric == "l1":
                                results["Single_run"]["Distance_between_best_and_intermediate_best_cfes"] = l1_distance_between_best_and_intermediate_best_cfes
                                results["Multiple_runs"]["Avg_distance_between_best_and_intermediate_best_cfes"] += l1_distance_between_best_and_intermediate_best_cfes
                            elif self.distance_metric == "weighted_l1":
                                results["Single_run"]["Distance_between_best_and_intermediate_best_cfes"] = weighted_l1_distance_between_best_and_intermediate_best_cfes
                                results["Multiple_runs"]["Avg_distance_between_best_and_intermediate_best_cfes"] += weighted_l1_distance_between_best_and_intermediate_best_cfes
                            
                            results["Single_run"]["L2_distance_between_best_and_intermediate_best_cfes"] = l2_distance_between_best_and_intermediate_best_cfes
                            results["Single_run"]["L1_distance_between_best_and_intermediate_best_cfes"] = l1_distance_between_best_and_intermediate_best_cfes
                            results["Single_run"]["Weighted_L1_distance_between_best_and_intermediate_best_cfes"] = weighted_l1_distance_between_best_and_intermediate_best_cfes
                            
                            results["Multiple_runs"]["Avg_l2_distance_between_best_and_intermediate_best_cfes"] += l2_distance_between_best_and_intermediate_best_cfes
                            results["Multiple_runs"]["Avg_l1_distance_between_best_and_intermediate_best_cfes"] += l1_distance_between_best_and_intermediate_best_cfes
                            results["Multiple_runs"]["Avg_weighted_l1_distance_between_best_and_intermediate_best_cfes"] += weighted_l1_distance_between_best_and_intermediate_best_cfes
                        else:
                            empty_intermediate_counter += 1

                        results["Single_run"]["Time_dynamic"] = time_intermediate_dynamic
                        results["Multiple_runs"]["Avg_Time_dynamic"] += results["Single_run"]["Time_dynamic"] 
                
                if self.dynamic_constraints:
                    self.constraints, self.immutables = {}, []
                    self.setup_constraints()
            if self.initial_population_strategy != "kdtree":
                results_X["population_generation_time"] = np.median(population_generation_time_per_run)
            else:
                results_X["population_generation_time"] = population_generation_time_global
            results_X["prepare_instances_time"] = prepare_instances_time

            if running_times_per_instance > 1:
                results["Multiple_runs"]["Avg_Best_cfe_distance"] = safe_divide(results["Multiple_runs"]["Avg_Best_cfe_distance"], results["Multiple_runs"]["Times_that_at_least_one_cfe_found_percentage"])
                results["Multiple_runs"]["Avg_Best_cfe_l2_distance"] = safe_divide(results["Multiple_runs"]["Avg_Best_cfe_l2_distance"], results["Multiple_runs"]["Times_that_at_least_one_cfe_found_percentage"])
                results["Multiple_runs"]["Avg_Best_cfe_l1_distance"] = safe_divide(results["Multiple_runs"]["Avg_Best_cfe_l1_distance"], results["Multiple_runs"]["Times_that_at_least_one_cfe_found_percentage"])
                results["Multiple_runs"]["Avg_Best_cfe_weighted_l1_distance"] = safe_divide(results["Multiple_runs"]["Avg_Best_cfe_weighted_l1_distance"], results["Multiple_runs"]["Times_that_at_least_one_cfe_found_percentage"])
                results["Multiple_runs"]["Avg_Best_cfe_sparsity"] = safe_divide(results["Multiple_runs"]["Avg_Best_cfe_distance"], results["Multiple_runs"]["Times_that_at_least_one_cfe_found_percentage"])

                results["Multiple_runs"]["Avg_Intermediate_best_cfe_distance"] = safe_divide(results["Multiple_runs"]["Avg_Intermediate_best_cfe_distance"], results["Multiple_runs"]["Times_that_at_least_one_intermediate_cfe_found_percentage"])
                results["Multiple_runs"]["Avg_Intermediate_best_cfe_l2_distance"] = safe_divide(results["Multiple_runs"]["Avg_Intermediate_best_cfe_l2_distance"], results["Multiple_runs"]["Times_that_at_least_one_intermediate_cfe_found_percentage"])
                results["Multiple_runs"]["Avg_Intermediate_best_cfe_l1_distance"] = safe_divide(results["Multiple_runs"]["Avg_Intermediate_best_cfe_l1_distance"], results["Multiple_runs"]["Times_that_at_least_one_intermediate_cfe_found_percentage"])
                results["Multiple_runs"]["Avg_Intermediate_best_cfe_weighted_l1_distance"] = safe_divide(results["Multiple_runs"]["Avg_Intermediate_best_cfe_weighted_l1_distance"], results["Multiple_runs"]["Times_that_at_least_one_intermediate_cfe_found_percentage"])
                results["Multiple_runs"]["Avg_Intermediate_best_cfe_sparsity"] = safe_divide(results["Multiple_runs"]["Avg_Intermediate_best_cfe_sparsity"], results["Multiple_runs"]["Times_that_at_least_one_intermediate_cfe_found_percentage"])
                
                results["Multiple_runs"]["Avg_number_of_generations"] /= running_times_per_instance
                results["Multiple_runs"]["Avg_Time_dynamic"] /= running_times_per_instance

                results["Multiple_runs"]["Avg_distance_between_best_and_intermediate_best_cfes"] = safe_divide(results["Multiple_runs"]["Avg_distance_between_best_and_intermediate_best_cfes"], results["Multiple_runs"]["Times_that_at_least_one_intermediate_cfe_found_percentage"])

            results_X[i] = results
            iteration += 1
        
        if self.dynamic_constraints:
            print(f"Empty intermediate counter: {empty_intermediate_counter}")
        return results_X

    def l2_distance(self, *args):
        if len(args) == 1:
            x_prime = args[0]
            return np.linalg.norm(np.array(self.normalized_x_numpy) - np.array(x_prime))
        elif len(args) == 2:
            a, b = args
            return np.linalg.norm(np.array(a) - np.array(b).reshape(1,-1), axis=1)
        else:
            raise ValueError("Invalid number of arguments. Provide 1 or 2 arguments.")
    
    def normalized_l2_distance(self, *args):
        if len(args) == 1:
            x_prime = args[0]
            return np.linalg.norm(np.array(self.normalized_x_numpy) - np.array(x_prime)) / self.max_l2_distance
        elif len(args) == 2:
            a, b = args
            return np.linalg.norm(np.array(a) - np.array(b).reshape(1,-1), axis=1) / self.max_l2_distance
        else:
            raise ValueError("Invalid number of arguments. Provide 1 or 2 arguments.")

    def l1_distance(self, *args):
        if len(args) == 1:
            x_prime = args[0]
            return np.linalg.norm(np.array(self.normalized_x_numpy) - np.array(x_prime), ord=1)
        elif len(args) == 2:
            a, b = args
            a = np.array(a)
            b = np.array(b)
            if b.ndim == 1:
                b = b.reshape(1, -1)
            return np.sum(np.abs(a - b), axis=1)
        else:
            raise ValueError("Invalid number of arguments. Provide 1 or 2 arguments.")
        
    def normalized_l1_distance(self, *args):
        if len(args) == 1:
            x_prime = args[0]
            return np.linalg.norm(np.array(self.normalized_x_numpy) - np.array(x_prime), ord=1) / self.max_l1_distance
        elif len(args) == 2:
            a, b = args
            a = np.array(a)
            b = np.array(b)
            if b.ndim == 1:
                b = b.reshape(1, -1)
            return np.sum(np.abs(a - b), axis=1) / self.max_l1_distance
        else:
            raise ValueError("Invalid number of arguments. Provide 1 or 2 arguments.")
    
    def compute_proximity_loss_dice(self, *args):
        """Compute weighted distance between two vectors."""
        genes = None
        if len(args) == 1:
            genes = args[0]
            query_instance_normalized = self.normalized_x_numpy
        elif len(args) == 2:
            # this works between the incremental best, intermediate best cfe
            genes, query_instance_normalized = args 
        else:
            raise ValueError("Expected 1 or 2 arguments, got {}".format(len(args)))

        feature_weights = np.array(
            [self.feature_weights_list[0][i] for i in self.numerical_feature_indexes]
        )
        if genes.ndim == 1:
            genes = genes.reshape(1, -1)
        product = np.multiply(
            abs(genes[:, self.numerical_feature_indexes] - query_instance_normalized[self.numerical_feature_indexes]),
            feature_weights
        )
        return np.sum(product, axis=1) / sum(feature_weights)

    def sparsity(self, *args):
        """Compute weighted sparsity between two vectors."""
        if len(args) == 1:
            x_prime = args[0]
            return np.count_nonzero(np.array(self.normalized_x_numpy) - np.array(x_prime))
        elif len(args) == 2:
            a, b = args
            return np.count_nonzero(np.array(a) - np.array(b).reshape(1,-1), axis=1)
        else:
            raise ValueError("Invalid number of arguments. Provide 1 or 2 arguments.")
        
    def normalized_sparsity(self, *args):
        """Compute weighted sparsity between two vectors."""
        if len(args) == 1:
            x_prime = args[0]
            return np.count_nonzero(np.array(self.normalized_x_numpy) - np.array(x_prime)) / self.max_sparsity
        elif len(args) == 2:
            a, b = args
            return np.count_nonzero(np.array(a) - np.array(b).reshape(1,-1), axis=1) / self.max_sparsity
        else:
            raise ValueError("Invalid number of arguments. Provide 1 or 2 arguments.")

    def violation(self, x_prime):
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
                immutable_score += 1000 if x_prime[attribute_index] == self.x_numpy[attribute_index] else -1000
            elif attribute_index in self.constraints:
                feature = self.feature_names[attribute_index]
                lower, upper = self.constraints[attribute_index]
                if lower <= x_prime[attribute_index] <= upper:
                    ranges_score += 1000
                else:
                    scaler = self.min_max_scaler_per_column[feature]
                    lower_scaled_bound, upper_scaled_bound = scaler.transform(
                        np.array([[lower], [upper]]).reshape(-1, 1)).flatten()

                    x_prime_scaled_value = scaler.transform(
                        np.array([x_prime[attribute_index]]).reshape(-1, 1))[0][0]

                    ranges_score -= 1000 * abs(x_prime_scaled_value - lower_scaled_bound) if x_prime_scaled_value < lower_scaled_bound else abs(x_prime_scaled_value - upper_scaled_bound)

        return immutable_score, ranges_score
       
    def generate_individual(self):
        """
        Generate an individual by considering constraints, immutability, and categorical/numerical ranges.
        :return: An individual
        """        
        genes = []
        for i in range(len(self.x_label_encoded_numpy)):
            skip_feature_change_prob = random.random()
            feature_name = self.feature_names[i]

            # Randomly decide whether to change this feature or not
            if skip_feature_change_prob > self.initial_population_variability:
                genes.append(self.x_label_encoded_numpy[i])
                continue
            
            # If the feature is immutable, keep its original value
            if i in self.immutables:
                genes.append(self.x_label_encoded_numpy[i])
                continue                

            # For the categorical features, randomly select one of the values
            elif feature_name in self.categorical_columns:
                if self.constraints.get(i):
                    possible_values = self.constraints[i].split()
                    possible_values = [possible_values[i] for i in range(len(possible_values))]
                    genes.append(random.choice(possible_values))
                else:
                    genes.append(random.choice(self.features_ranges[feature_name]))
            
            # For numerical features, generate values within the defined constraints (if provided) or range
            else:
                if self.constraints.get(i):
                    if isinstance(self.constraints[i], str):
                        if self.constraints[i] == "inc":
                            upper_bound = self.features_ranges[feature_name][1]
                            if self.features_type[feature_name] == 'int':
                                genes.append(random.randint(self.x_label_encoded_numpy[i], upper_bound))
                            else:
                                genes.append(random.uniform(self.x_label_encoded_numpy[i], upper_bound))
                        elif self.constraints[i] == "dec":
                            lower_bound = self.features_ranges[feature_name][0]
                            if self.features_type[feature_name] == 'int':
                                genes.append(random.randint(lower_bound, self.x_label_encoded_numpy[i]))
                            else:
                                genes.append(random.uniform(lower_bound, self.x_label_encoded_numpy[i]))
                    else:
                        value = self.constraints[i].split()
                        lower, upper = int(value[0]), int(value[1])

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
        num_features = len(self.feature_names)
        crossover_points = sorted(random.sample(range(1, num_features), crossover_points))

        offspring1 = np.zeros(num_features, dtype=np.float64)
        offspring2 = np.zeros(num_features, dtype=np.float64)

        current_parent_idx = random.randint(0, num_parents - 1)
        start_idx = 0
        for point in crossover_points:
            for i in range(start_idx, point):
                offspring1[i] = parents[current_parent_idx].genes[i]
                offspring2[i] = parents[(current_parent_idx + 1) % num_parents].genes[i]
            current_parent_idx = (current_parent_idx + 1) % num_parents
            start_idx = point

        for i in range(start_idx, num_features):
            offspring1[i] = parents[current_parent_idx].genes[i]
            offspring2[i] = parents[(current_parent_idx + 1) % num_parents].genes[i]

        return Individual(offspring1), Individual(offspring2)

    def mutate_individual(self, individual):
        skip_mutation_indexes = set()

        for i in range(len(self.feature_names)):
            if i in skip_mutation_indexes:
                continue

            if random.random() < self.mutpb:
                feature_name = self.feature_names[i]

                # Handle immutables
                if i in self.immutables:
                    if self.x_label_encoded_numpy[i] != individual.genes[i]:
                        individual.genes[i] = self.x_label_encoded_numpy[i]
                    continue

                original_value = individual.genes[i]
                lower, upper = self.constraints.get(i, self.features_ranges[feature_name])

                # Validate against data distribution if needed
                if self.data_distribution:
                    lower_data_distribution, upper_data_distribution = self.features_ranges[feature_name]
                    if lower < lower_data_distribution or upper > upper_data_distribution:
                        print(f"Constraint for {feature_name} [{lower}, {upper}] violates data distribution [{lower_data_distribution}, {upper_data_distribution}]")
                        sys.exit()

                new_value = original_value

                if self.features_type[feature_name] == 'int':
                    possible_values = list(set(range(int(np.ceil(lower)), int(np.floor(upper)) + 1)) - {int(original_value)})
                    if possible_values:
                        new_value = random.choice(possible_values)
                else:
                    if lower == upper:
                        new_value = lower
                    else:
                        while True:
                            candidate = random.uniform(lower, upper)
                            if candidate != original_value:
                                new_value = candidate
                                break
                individual.genes[i] = new_value
        return individual
    
    def fitness_assignment_population(self, population, clear_fitness=False):
        if clear_fitness:
            for ind in population:
                ind.fitness = None

        genes = [ind.genes for ind in population]
        decoded_genes = decode_label_encoded_data(deepcopy(genes), self.feature_names, self.categorical_columns, self.categorical_label_encoders)
        if len(decoded_genes) == 0:
            print(f"Empty decoded genes: {decoded_genes}")
        ## turn the decoded genes into dataframes for predictions
        decoded_genes = pd.DataFrame(decoded_genes, columns=self.feature_names)
        normalized_genes = normalize_data(genes, self.feature_names, self.dataset)

        distances = None
        if self.distance_metric == "l2":
            distances = self.normalized_l2_distance(normalized_genes, self.normalized_x_numpy)
        elif self.distance_metric == "l1":
            distances = self.normalized_l1_distance(normalized_genes, self.normalized_x_numpy)
        elif self.distance_metric == "weighted_l1":
            distances = self.compute_proximity_loss_dice(normalized_genes, self.normalized_x_numpy)
        else: 
            raise ValueError("Invalid distance metric. Supported metrics are: l1, l2, weighted_l1")

        sparsities = self.normalized_sparsity(normalized_genes, self.normalized_x_numpy)
        y_primes = f_model(decoded_genes, self.model)

        y_scores = None
        if self.normalized_fitness:
            y_scores = np.where(y_primes == self.original_prediction, -1, 1)
        else:
            y_scores = np.where(y_primes == self.original_prediction, 0, 100)

        fitness_scores = -self.lambda1 * distances - self.lambda2 * sparsities + self.lambda3 * y_scores

        for ind, fitness in zip(population, fitness_scores):
            ind.fitness = fitness
        return population, max(fitness_scores)
    
    def get_closest_positive_instances(self, population_size=100):
        """
        Find the closest positive instances to a given negative instance.
        """
        population_around_instance_time = time()
        # Query for nearest neighbors of the first negative instance
        distances, indices = self.kdTree.query(self.x_label_encoded, k=len(self.kdTree_dataset))

        # Exclude the first neighbor (self)
        distances, indices = distances[:, 1:], indices[:, 1:]
        neighbors = zip(indices[0], distances[0])

        # Filter neighbors for positives and use a heap to keep the closest `num_positives`
        closest_positives = heapq.nsmallest(
            population_size,
            ((idx, dist) for idx, dist in neighbors if idx in self.positive_indices),
            key=lambda x: x[1]
        )

        # Get indices of the closest positive neighbors
        closest_positive_indices = [idx for idx, _ in closest_positives]
        closest_positive_instances = self.dataset.loc[closest_positive_indices]
        return [Individual(genes=row.tolist()) for _, row in closest_positive_instances.iterrows()], population_around_instance_time

    def initialize_population(self, population_size=100, initial_population_strategy="kdtree"):
        if initial_population_strategy == "kdtree":
            closest_positive_instances, population_around_instance_time =\
                self.get_closest_positive_instances(population_size)
            population = self.update_violators(closest_positive_instances)
            return population, 0, (time() - population_around_instance_time)
        
        elif initial_population_strategy == "random":
            population = []
            population_around_instance_time = time()
            unique_individuals = set()
            while len(population) < population_size:
                new_individual = self.generate_individual()
                genes_tuple = tuple(new_individual.genes)
                if genes_tuple not in unique_individuals:
                    population.append(new_individual)
                    unique_individuals.add(genes_tuple)
            return population, 0, (time() - population_around_instance_time) 
        else:
            sys.exit(f"Invalid initial population strategy: {self.initial_population_strategy}.")

    def check_null_updated_constraints(self):
        for key, value in self.updated_constraints.items():
            if value != "":
                return False
        return True

    def evolve(self):
        time_intermediate_from_scratch = 0
        generations_from_scratch = 0
        population_generation_time = None
        
        def generations(population, num_generations, best_fitness):          
            elite_count = max(1, int(self.elite_ratio * len(population)))
            gen = 0
            previous_best_fitness = best_fitness
            current_best_fitness = float("-inf")
            generations_without_improvement = 0

            # Termination Criteria
            while gen < num_generations:
                if abs(current_best_fitness - previous_best_fitness) < self.thresh:
                    generations_without_improvement += 1
                else:
                    generations_without_improvement = 0

                if generations_without_improvement >= self.early_stopping_iterations:
                    break
                previous_best_fitness = current_best_fitness

                # Elitism
                if self.elite_ratio > 0:
                    elite_individuals = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:elite_count]

                # Parent Selection
                if self.selection_method == "tournament":
                    parents = self.tournament_selection(population, self.num_parents)
                elif self.selection_method == "roulette":
                    parents = self.roulette_selection(population, self.num_parents)
                elif self.selection_method == "rank":
                    parents = self.rank_selection(population, self.num_parents)
                elif self.selection_method == "sus":
                    parents = self.sus_selection(population, self.num_parents)
                elif self.selection_method == "50percentbest":
                    parents = self.fifty_percent_best_selection(population, self.num_parents)
                
                offspring = []
                offspring_size = len(population) - elite_count

                # Create offspring until the required population size is reached
                while len(offspring) < offspring_size:
                    # Select parent pairs for crossover, allowing repetition
                    # parent1, parent2 = deepcopy(random.choice(parents)), deepcopy(random.choice(parents))
                    try:
                        # Try selecting two distinct parents
                        parent1, parent2 = deepcopy(random.sample(parents, 2))
                        if np.array_equal(parent1.genes, parent2.genes):
                            raise ValueError("Identical parents")
                    except:
                        # If unable to find distinct parents, fall back to cloning any two and rely on mutation
                        parent1 = parent2 = deepcopy(random.choice(parents))

                    if random.random() < self.cxpb:
                        child1, child2 = self.crossover([parent1, parent2], self.crossover_points)
                        offspring.extend([child1, child2])
                    else:
                        # No crossover, directly clone
                        offspring.extend([parent1, parent2])

                for mutant in offspring:
                    if random.random() < self.mutpb:
                        self.mutate_individual(mutant)

                # Ensure only the needed number of offspring are retained
                if self.elite_ratio > 0 and len(offspring) > offspring_size:
                    # Keep the best individuals from the offspring (offspring_size)
                    # Then Combine elite individuals with the new offspring to form the next generation
                    offspring, _ = self.fitness_assignment_population(offspring)
                    population = elite_individuals + sorted(offspring, key=lambda ind: ind.fitness, reverse=True)[:offspring_size]

                genes_seen = set()
                unique_population = []
                for ind in population:
                    gene_tuple = tuple(ind.genes)
                    if gene_tuple not in genes_seen:
                        genes_seen.add(gene_tuple)
                        unique_population.append(ind)
                population = unique_population
                if len(population) == 1:
                    # add the same individual to the population again, after mutation
                    population.append(self.mutate_individual(population[0]))
                elif len(population) == 0:
                    print("Empty population")
                
                population, current_best_fitness = self.fitness_assignment_population(population, clear_fitness=True)
                gen += 1
                if self.reweighting_lambdas_after_generations >= 0 and gen == self.reweighting_lambdas_after_generations:
                    self.lambda1 = self.lambdas_reweighting_after_updating_constraints_and_some_generations["lambda1"]
                    self.lambda2 = self.lambdas_reweighting_after_updating_constraints_and_some_generations["lambda2"]
                    self.lambda3 = self.lambdas_reweighting_after_updating_constraints_and_some_generations["lambda3"]
            return population, gen
        
        regeneration_tries_intermediate = -1
        population = []
        unique_applicable_cfes_from_scratch = []
        unique_applicable_cfes_to_unique_individuals_percentage_from_scratch = 0
        solution = []

        while regeneration_tries_intermediate <= self.regeneration_tries_intermediate:
            population, population_generation_time, population_around_instance_time = self.initialize_population(self.population_size)
            population_generation_time += population_generation_time
            time_intermediate_from_scratch += population_around_instance_time

            genetic_algorithm_time = time()
            population, best_fitness = self.fitness_assignment_population(population)
            population, generations_from_scratch = generations(population, self.num_generations, best_fitness)
            
            unique_applicable_cfes_from_scratch, unique_applicable_cfes_to_unique_individuals_percentage_from_scratch =\
                self.changed_prediction_individuals(population)

            time_intermediate_from_scratch += time() - genetic_algorithm_time
            if len(unique_applicable_cfes_from_scratch) > 0:
                solution = unique_applicable_cfes_from_scratch[0]
                break
            else:
                regeneration_tries_intermediate += 1
        if self.verbose:
            if len(solution) == 0:
                best_individual = max(population, key=lambda ind: ind.fitness)
                print(f"No solution found. Best individual: {best_individual.genes})")
            else:
                pass
                # cfe_reshaped = solution.reshape(1, -1)
                # inverse_transformed_cfe_reshaped_df = inverse_transform_dataset_with_onehot(cfe_reshaped, self.model, self.numerical_columns, self.binary_categorical_columns, self.feature_names[self.numerical_and_binary_categorical_len:]).iloc[0]
                # inverse_transformed_cfe_reshaped_features = inverse_transformed_cfe_reshaped_df.to_dict()
                # display_cfe_comparison(self.x_dict, inverse_transformed_cfe_reshaped_features)

        # If dynamic constraints are enabled:
        #   ask the user for acceptance
        #   and update constraints
        self.population = population
        generations_dynamic = 0
        if self.dynamic_constraints:
            # if dynamic, but not updated constraints.
            if self.check_null_updated_constraints() and self.automatic_user_acceptance:
                return unique_applicable_cfes_from_scratch, unique_applicable_cfes_from_scratch, 0, time_intermediate_from_scratch, 0, 0, unique_applicable_cfes_to_unique_individuals_percentage_from_scratch
                
            unique_applicable_cfes = []
            time_dynamic = 0
            unique_applicable_cfes_to_unique_individuals_percentage = 0
            if not self.multistep_updated_constraints:
                self.get_updated_constraints(values=None)

                if solution == []:
                    accepted = 'n'
                else:
                    accepted = self.ask_user_acceptance(best_individual=solution)
                solution = []

                regeneration_tries_intermediate_after_updating_constraints = -1
                
                while accepted != 'y' and regeneration_tries_intermediate_after_updating_constraints\
                                    <= self.regeneration_tries_intermediate_after_updating_constraints:
                    start_time_intermediate_dynamic = time()
                    best_fitness = float("-inf")
                    ##### NEW RANDOM POPULATION #####
                    if self.strategy=="new_random_population":
                        population, _, _ = self.initialize_population(population_size=self.population_size, initial_population_strategy="random")
                        population, best_fitness = self.fitness_assignment_population(population, clear_fitness=True)
                    ##### UPDATE FITNESS #####
                    elif self.strategy == "update_fitness":
                        population, best_fitness = self.fitness_assignment_population(population, clear_fitness=True)
                    ##### FIXING THE POPULATION #####
                    elif self.strategy == "fix_population_update_fitness":
                        population = self.update_violators(population)
                        population, best_fitness = self.fitness_assignment_population(population, clear_fitness=True)
                    else:
                        raise ValueError("Invalid strategy. Supported strategies are: new_random_population, update_fitness, fix_population_update_fitness")
                    ##### ADD INDIVIDUALS THAT ADHERE TO THE NEW CONSTRAINTS TO THE POPULATION #####
                    if self.strategy == "Add_individuals_with_respected_constraints_n_keep_the_best":
                        if self.population_size_dynamic > 0:
                            new_population = self.initialize_population(population_size=self.population_size_dynamic)
                            population.extend(new_population)
                            population, best_fitness = self.fitness_assignment_population(population, clear_fitness=True)
                            population = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:-self.population_size_dynamic]
                        else:
                            print("The population size dynamic should be greater than 0 for this strategy.")
                            sys.exit()

                    ### APPLY GENERATIONS TO THE CURRENT POPULATION AFTER PROVIDING THE CURRENT MAX FITNESS VALUE ###
                    population, generations_dynamic = generations(population, self.num_generations, best_fitness)
                    unique_applicable_cfes, unique_applicable_cfes_to_unique_individuals_percentage = self.changed_prediction_individuals(population)
                    time_dynamic += (time() - start_time_intermediate_dynamic)

                    if len(unique_applicable_cfes) > 0:
                        solution = unique_applicable_cfes[0]
                        accepted = self.ask_user_acceptance(best_individual=solution)
                    regeneration_tries_intermediate_after_updating_constraints += 1

                if self.verbose:
                    if len(unique_applicable_cfes) < 0:
                        best_individual = max(population, key=lambda ind: ind.fitness)
                        print(f"No solution found. Best individual: {best_individual.genes})")
                    else:
                        pass
                        # get the best cfe:
                        # cfe = unique_applicable_cfes[0]
                        # # inverse_transformed_cfe_reshaped_df = inverse_transform_dataset_with_onehot(cfe_reshaped, self.model, self.numerical_columns, self.binary_categorical_columns, self.feature_names[self.numerical_and_binary_categorical_len:]).iloc[0]
                        # inverse_transformed_cfe_reshaped_features = inverse_transformed_cfe_reshaped_df.to_dict()
                        # display_cfe_comparison(self.x_dict, inverse_transformed_cfe_reshaped_features)

                self.population = population
            else:
                for key, values in self.updated_constraints.items():
                    self.get_updated_constraints(values=values)
                    # In case of solution is found and automatic assesment is enabled, to continue adding constraints:
                    #       the user hypothetically does not accept the solution
                    if self.automatic_user_acceptance:
                        accepted = 'n'
                    else:
                        if solution == []:
                            accepted = 'n'
                        else:
                            accepted = self.ask_user_acceptance(best_individual=solution)
                    regeneration_tries_intermediate_after_updating_constraints = -1
                    
                    while accepted != 'y' and regeneration_tries_intermediate_after_updating_constraints\
                                        != self.regeneration_tries_intermediate_after_updating_constraints:
                        start_time_intermediate_dynamic = time()
                        best_fitness = float("-inf")
                        ##### NEW RANDOM POPULATION #####
                        if self.strategy == "new_random_population":
                            population, _, _ = self.initialize_population(population_size=self.population_size, initial_population_strategy="random")
                            population, best_fitness = self.fitness_assignment_population(population, clear_fitness=True)
                        ##### FIXING THE POPULATION #####
                        elif self.strategy == "fix_population_update_fitness":
                            population = self.update_violators(population)
                            population, best_fitness = self.fitness_assignment_population(population, clear_fitness=True)
                        ##### ADD INDIVIDUALS THAT ADHERE TO THE NEW CONSTRAINTS TO THE POPULATION #####
                        if self.strategy == "Add_individuals_with_respected_constraints_n_keep_the_best":
                            if self.population_size_dynamic > 0:
                                new_population = self.initialize_population(population_size=self.population_size_dynamic)
                                population.extend(new_population)
                                population, best_fitness = self.fitness_assignment_population(population, clear_fitness=True)
                                population = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:-self.population_size_dynamic]
                            else:
                                print("The population size dynamic should be greater than 0 for this strategy.")
                                sys.exit()

                        ### APPLY GENERATIONS TO THE CURRENT POPULATION AFTER PROVIDING THE CURRENT MAX FITNESS VALUE ###
                        self.population, generations_dynamic_ = generations(self.population, self.num_generations, best_fitness)
                        generations_dynamic += generations_dynamic_
                        unique_applicable_cfes, unique_applicable_cfes_to_unique_individuals_percentage = self.changed_prediction_individuals(self.population)
                        time_dynamic += (time() - start_time_intermediate_dynamic)
                        if len(unique_applicable_cfes) > 0:
                            solution = unique_applicable_cfes[0]
                            accepted = self.ask_user_acceptance(best_individual=solution)
                        regeneration_tries_intermediate_after_updating_constraints += 1
                    if self.verbose:
                        if len(unique_applicable_cfes) < 0:
                            best_individual = max(population, key=lambda ind: ind.fitness)
                            print(f"No solution found. Best individual: {best_individual.genes})")
                        else:
                            pass
                            # get the best cfe:
                            # cfe = unique_applicable_cfes[0]
                            # # inverse_transformed_cfe_reshaped_df = inverse_transform_dataset_with_onehot(cfe_reshaped, self.model, self.numerical_columns, self.binary_categorical_columns, self.feature_names[self.numerical_and_binary_categorical_len:]).iloc[0]
                            # inverse_transformed_cfe_reshaped_features = inverse_transformed_cfe_reshaped_df.to_dict()
                            # display_cfe_comparison(self.x_dict, inverse_transformed_cfe_reshaped_features)

                    # self.population = population
            return unique_applicable_cfes_from_scratch, unique_applicable_cfes, population_generation_time, time_intermediate_from_scratch, time_dynamic, generations_dynamic, unique_applicable_cfes_to_unique_individuals_percentage
        else:
            self.population = population
            return None, unique_applicable_cfes_from_scratch, population_generation_time, time_intermediate_from_scratch,\
                  None, generations_from_scratch, unique_applicable_cfes_to_unique_individuals_percentage_from_scratch
    
    def update_violators(self, population):
        """
        Updates violators in the population by adjusting genes that do not satisfy constraints or immutability.
        """
        population = deepcopy(population)
        
        # Precompute immutable and constraint-related checks
        immutable_set = set(self.immutables)
        constraints_keys = set(self.constraints.keys())
        for ind in population:
            for i, (gene, ground_point) in enumerate(zip(ind.genes, self.x_label_encoded_numpy)):
                if gene == ground_point:
                    continue
                
                feature_name = self.feature_names[i]
                # Handle immutable features
                if i in immutable_set:
                    ind.genes[i] = ground_point
                    continue
                
                # Handle constraints for numerical features
                elif i in constraints_keys:
                    if isinstance(self.constraints[i], str):
                        if self.constraints[i] == "incr":
                            if gene > ground_point:
                                continue
                            else:
                                # set a random value from the ground point to the upper bound
                                upper_bound = self.features_ranges[feature_name][1]
                                if self.features_type[feature_name] == 'int':
                                    ind.genes[i] = random.randint(ground_point, upper_bound)
                                else:
                                    ind.genes[i] = random.uniform(ground_point, upper_bound)                                
                        elif self.constraints[i] == "decr":
                            if gene < ground_point:
                                continue
                            else:
                                # set a random value from the lower bound to the ground point
                                lower_bound = self.features_ranges[feature_name][0]
                                if self.features_type[feature_name] == 'int':
                                    ind.genes[i] = random.randint(lower_bound, ground_point)
                                else:
                                    ind.genes[i] = random.uniform(lower_bound, ground_point)
                    lower, upper = self.constraints[i]
                    if not (lower < gene < upper):
                        # Replace with random value within constraints
                        if self.features_type[feature_name] == 'int':
                            ind.genes[i] = random.randint(lower, upper)
                        else:
                            ind.genes[i] = random.uniform(lower, upper)
        return population

    def changed_prediction_individuals(self, population):
        """
        Find the unique applicable CFEs, count them, and calculate the percentage 
        compared to the unique individuals in a population.

        - Returns individuals in their original (label-encoded) form.
        - Uses decoded versions only for model predictions.
        """
        unique_individuals = {tuple(ind.genes): ind for ind in population}

        decoded_genes = decode_label_encoded_data(
            [ind.genes for ind in unique_individuals.values()], 
            self.feature_names, self.categorical_columns, self.categorical_label_encoders
        )
        decoded_df = pd.DataFrame(decoded_genes, columns=self.feature_names)

        # Find individuals that change the prediction using decoded genes
        unique_applicable_cfes = [
            individual for ((genes, individual), (_, decoded_instance)) in zip(unique_individuals.items(), decoded_df.iterrows()) 
            if f_model(decoded_instance.to_frame().T, self.model) != self.original_prediction
        ]
        return unique_applicable_cfes, (len(unique_applicable_cfes) / len(unique_individuals)) * 100

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
        counts = Counter(individual_tuples)
        identical_count = sum(count for count in counts.values() if count > 1)
        score = (identical_count / len(population)) * 100
        return score
        
    def max_fitness(self, population):
        return max([ind.fitness for ind in population])
    
    def ask_user_acceptance(self, best_individual=None):   
        """
        Ask the user for acceptance of the counterfactual explanation.
        :param best_individual: The best individual in the population
        :return: True if the user accepts the counterfactual explanation, False otherwise"""
        while True:
            if not self.automatic_user_acceptance:
                user_response = input("Do you accept this counterfactual explanation? (y/n): ").strip().lower()
            else:
                user_response = self.check_constraints(best_individual)
            if user_response in ["y", "n"]:
                return user_response
            else:
                print("Invalid input. Please type 'y' or 'n'.")

    def check_constraints(self, best_individual):
        """
        Check if the best individual adheres to the constraints and immutability.
        
        - Parameters:
            - best_individual: The best individual in the population

        - Returns:
            - 'y' if the best individual adheres to the constraints and immutability
            - 'n' otherwise
        """
        for i in self.immutables:
            if best_individual.genes[i] != self.x_label_encoded_numpy[i]:
                return 'n'
            elif self.x_label_encoded_numpy[i] != best_individual.genes[i]:
                return 'n'
        for i in self.constraints:
            if isinstance(self.constraints[i], str):
                if self.constraints[i] == "incr":
                    if not (best_individual.genes[i] > self.x_label_encoded_numpy[i]):
                        return 'n'
                elif self.constraints[i] == "decr":
                    if not (best_individual.genes[i] < self.x_label_encoded_numpy[i]):
                        return 'n'
            else:
                lower, upper = self.constraints[i]
                if not (lower <= best_individual.genes[i] <= upper):
                    return 'n'
        return 'y'

    # Mock function to get updated constraints from the user in the original space
    def get_updated_constraints(self):
        if self.verbose:
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
                        if self.verbose:
                            print(f"Feature '{base_feature_name}' is passed (no new constraints).")
                        break
                    elif user_input == "i":
                        for idx in find_indices:
                            if idx not in self.immutables:
                                self.immutables.append(idx)
                        if self.verbose:
                            print(f"Feature '{base_feature_name}' marked as immutable.")
                        break
                    elif user_input == "ni":
                        ## remove the immutable constraint
                        ## pop the skip indices from the self.immutables
                        for idx in find_indices:
                            self.immutables.remove(idx)                        
                        if self.verbose:
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
                        if self.verbose:
                            print(f"Feature '{feature}' is passed (no new constraints).")
                        break
                    
                    # User enters '-' to mark the feature as immutable
                    elif user_input == "i":
                        self.immutables.append(i)
                        ## if the feature is in the constraints then remove it
                        if i in self.constraints:
                            self.constraints.pop(i)
                            if self.verbose:
                                print(f"Feature '{feature}' marked as immutable. Constraints removed.")
                        else:
                            if self.verbose:
                                print(f"Feature '{feature}' marked as immutable.")                        
                        break
                    
                    elif user_input == "ni":
                        if self.verbose:
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
                                    if self.verbose:
                                        print(f"Feature '{feature}' set to [{lower}, {upper}] and is no longer immutable.")
                                else:
                                    if self.verbose:
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