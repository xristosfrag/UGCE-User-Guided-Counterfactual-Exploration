import os
import numpy as np
import pandas as pd
from utils import label_encode_data, normalize_data
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

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

UGCE_dir = get_UGCE_directory()

################################### DICE-Baseline Evaluation ###################################
def compute_proximity_loss_dice(dice_object, a, b):
    """Compute weighted distance between two vectors."""
    genes, query_instance_normalized = a, b
    if query_instance_normalized.ndim == 2:
        query_instance_normalized = query_instance_normalized[0]
    
    feature_weights = np.array(
        [dice_object.feature_weights_list[0][i] for i in dice_object.numerical_feature_indexes]
    )

    if genes.ndim == 1:
        genes = genes.reshape(1, -1)

    product = np.multiply(
        np.abs(genes[:, dice_object.numerical_feature_indexes] - query_instance_normalized[dice_object.numerical_feature_indexes]),
        feature_weights
    )
    return np.sum(product, axis=1) / sum(feature_weights)

def aggregate_results_DICE_baseline_intermediate(dice_object, instances_to_explain, explainers_no_constraints, explainers, TARGET_COLUMN):
    avg_proximity_loss_iterations = []
    for i in range(5):
        dice_no_consts = explainers_no_constraints[i]
        dice_consts = explainers[i]
        avg_proximity_loss = 0
        encoded_instances, normalized_instances = prepare_instances(dice_object, instances_to_explain)
        index = 0
        found = 0
        for i, x in instances_to_explain.iterrows():
            try:
                cfes_no_constraints = dice_no_consts._cf_examples_list[index].final_cfs_df.drop(columns=[TARGET_COLUMN])
                cfes_constraints = dice_consts._cf_examples_list[index].final_cfs_df.drop(columns=[TARGET_COLUMN])

                if len(cfes_no_constraints) >= 1 and len(cfes_constraints) >= 1:
                    found += 1
                    cfe_constraints = cfes_constraints.iloc[0]
                    cfe_no_constraints = cfes_no_constraints.iloc[0]
                else:
                    continue
                cfe_constraints = cfe_constraints.to_frame().T
                encoded_cfee_constraints = label_encode_data(cfe_constraints, dice_object.feature_names, dice_object.categorical_columns,\
                                                            dice_object.categorical_label_encoders)
                normalized_cfe = normalize_data(encoded_cfee_constraints, dice_object.feature_names, dice_object.dataset)

                cfe_no_constraints = cfe_no_constraints.to_frame().T
                encoded_cfe_no_constraints = label_encode_data(cfe_no_constraints, dice_object.feature_names, dice_object.categorical_columns,\
                                                            dice_object.categorical_label_encoders)
                normalized_cfe_no_constraints = normalize_data(encoded_cfe_no_constraints, dice_object.feature_names, dice_object.dataset)
            except Exception as e:
                print(e)
                continue
            avg_proximity_loss += compute_proximity_loss_dice(dice_object, np.array(normalized_cfe_no_constraints), np.array(normalized_cfe))
        if found > 0:
            avg_proximity_loss /= found
        avg_proximity_loss_iterations.append(avg_proximity_loss)
    print_metric_stats("avg_proximity_loss", avg_proximity_loss_iterations)

def normalized_l2_distance(dice_obj, a, b):
    a = np.asarray(a, dtype=np.float64).reshape(1, -1)
    b = np.asarray(b, dtype=np.float64).reshape(1, -1)
    return np.linalg.norm(a - b, axis=1)[0] / dice_obj.max_l2_distance  # return scalar

def prepare_instances(dice_obj, instances_to_explain):
    encoded_instances = label_encode_data(instances_to_explain, dice_obj.feature_names, dice_obj.categorical_columns,\
                                                        dice_obj.categorical_label_encoders)
    normalized_instances = normalize_data(encoded_instances, dice_obj.feature_names, dice_obj.dataset)
    return encoded_instances, normalized_instances

def stats_DICE_baseline(dice_obj, instances_to_explain, dice_exp_genetic, TARGET_COLUMN):
    avg_distance = 0
    avg_l1 = 0
    avg_proximity_loss = 0
    avg_sparsity = 0
    time_genetic = 0
    build_tree_time = []
    prepare_query_instance_and_predict_time = 0
    found = 0
    generations = 0

    encoded_instances, normalized_instances = prepare_instances(dice_obj, instances_to_explain)
    index = 0
    for i, x in instances_to_explain.iterrows():
        try:
            encoded_x = instances_to_explain.loc[i].to_frame().T
            normalized_x = normalized_instances.loc[i].to_frame().T.to_numpy()[0]
            cfes = dice_exp_genetic._cf_examples_list[index].final_cfs_df.drop(columns=[TARGET_COLUMN])
            prepare_query_instance_and_predict_time += dice_exp_genetic._cf_examples_list[index].prepare_query_instance_and_predict_time
            build_tree_time.append(dice_exp_genetic._cf_examples_list[index].build_tree_time)
            time_genetic += dice_exp_genetic._cf_examples_list[index].time_genetic
            generations += dice_exp_genetic._cf_examples_list[index].generations
            if len(cfes) >= 1:
                found += 1
                cfe = cfes.iloc[0]
            else:
                continue
            encoded_cfe = cfe.to_frame().T
            encoded_cfe = label_encode_data(encoded_cfe, dice_obj.feature_names, dice_obj.categorical_columns,\
                                                        dice_obj.categorical_label_encoders)
            normalized_cfe = normalize_data(encoded_cfe, dice_obj.feature_names, dice_obj.dataset)
        except Exception as e:
            continue
        avg_distance += (normalized_l2_distance(dice_obj, normalized_x, normalized_cfe) / dice_obj.max_l2_distance)
        avg_l1 += (dice_obj.normalized_l1_distance(np.array(normalized_x), np.array(normalized_cfe)) / dice_obj.max_l1_distance)[0]
        avg_proximity_loss += dice_obj.compute_proximity_loss_dice(np.array(normalized_cfe), np.array(normalized_x))[0]
        avg_sparsity += (dice_obj.normalized_sparsity(np.array(normalized_x), np.array(normalized_cfe)) / dice_obj.max_sparsity)[0]
        index += 1
    
    avg_distance /= found 
    avg_sparsity /= found
    avg_l1 /= found
    generations /= found
    avg_proximity_loss /= found

    # if prepare_query_instance_and_predict_time > 60:
    #     print(f"Time taken for preparing query instance and predicting: {prepare_query_instance_and_predict_time/60} (minutes)")
    # else:
    #     print(f"Time taken for preparing query instance and predicting: {prepare_query_instance_and_predict_time} (seconds)")
    if time_genetic > 60:
        time_genetic = time_genetic/60
        # print(f"Time taken for generating counterfactuals using {method}: {time_genetic/60} (minutes)")
    # else:
    #     print(f"Time taken for generating counterfactuals using {method}: {time_genetic} (seconds)")
    full_time = time_genetic+prepare_query_instance_and_predict_time+np.median(build_tree_time)
    if full_time > 60:
        full_time = full_time/60
        # print(f"Full time taken for generating counterfactuals using {method}: {full_time} (minutes)")
    # else:
    #     print(f"Full time taken for generating counterfactuals using {method}: {full_time} (seconds)")

    # print(f"Cfes found: {found/len(instances_to_explain)*100}%")
    # print("Average distance between the explainee datapoints and the counterfactuals: ", avg_distance)
    # print("Average l1: ", avg_l1)
    # print("Average proximity loss: ", avg_proximity_loss)
    # print("Average sparsity: ", avg_sparsity)
    # print("Average generations: ", generations)    
    # print(f"Build tree time: {np.min(build_tree_time), np.median(build_tree_time), np.max(build_tree_time)}")
    return full_time, generations, found/len(instances_to_explain)*100, avg_distance, avg_l1, avg_proximity_loss, avg_sparsity

def aggregate_results_DICE_baseline(dice_object, instances_to_explain, dice_explainer, TARGET_COLUMN):
    # Placeholder for collecting results across all runs
    full_times = []
    coverages = []
    distances = []
    l1s = []
    proximities = []
    sparsities = []
    generation_counts = []

    # Run the evaluation loop
    for dice_explainer_instance in dice_explainer:
        dice_baseline_ful_time, dice_baseline_generations, dice_baseline_coverage, dice_baseline_avg_distance, dice_baseline_avg_l1, \
            dice_baseline_avg_proximity_loss, dice_baseline_avg_sparsity = stats_DICE_baseline(dice_object, instances_to_explain, dice_explainer_instance, TARGET_COLUMN)
        
        full_times.append(dice_baseline_ful_time)
        distances.append(dice_baseline_avg_distance)
        l1s.append(dice_baseline_avg_l1)
        proximities.append(dice_baseline_avg_proximity_loss)
        sparsities.append(dice_baseline_avg_sparsity)
        generation_counts.append(dice_baseline_generations)
        coverages.append(dice_baseline_coverage)

    # Compute and print means and stds
    def print_metric_stats(name, values):
        print(f"{name}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")

    print_metric_stats("Full Time", full_times)
    print_metric_stats("Generations", generation_counts)
    print_metric_stats("Coverage", coverages)
    # print_metric_stats("L2 Distance", distances)
    # print_metric_stats("L1 Distance", l1s)
    print_metric_stats("Proximity Loss", proximities)
    print_metric_stats("Sparsity", sparsities)

################################### UGCE-Baseline Evaluation ###################################
def print_metric_stats(name, values):
    print(f"{name}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")

def stats_baseline_intermediate(ugce_obj, results_baseline_explainer_no_constraints, results_baseline_explainer):
    avg_weighted_dist_from_intermediate_arr = []
    
    for i in range(5):
        results_baseline_explainer_no_constraints_local = results_baseline_explainer_no_constraints[i]
        results_baseline_explainer_local = results_baseline_explainer[i]
        avg_weighted_dist_from_intermediate = 0
        found = 0
        for instance_id, dictofit in results_baseline_explainer_no_constraints_local.items():
            if instance_id in ["population_generation_time", "prepare_instances_time"]:
                continue
            if instance_id not in results_baseline_explainer_local.keys() or len(np.array(results_baseline_explainer_local[instance_id]["Single_run"]["Normalized_Best_cfe"])) == 0:
                continue
            ## this is the distance of the baseline from the (no constraints) intermediate
            avg_weighted_dist_from_intermediate += \
                compute_proximity_loss_dice(ugce_obj, np.array(dictofit["Single_run"]["Normalized_Best_cfe"]),\
                                    np.array(results_baseline_explainer_local[instance_id]["Single_run"]["Normalized_Best_cfe"]))
            found += 1
                
        avg_weighted_dist_from_intermediate /= found
        avg_weighted_dist_from_intermediate_arr.append(avg_weighted_dist_from_intermediate)
    print_metric_stats("Avg weighted distance from intermediate", avg_weighted_dist_from_intermediate_arr)

def stats_baseline(ugce_obj, results_baseline):
    time_baseline = 0
    avg_l2 = 0
    avg_cfes_found = 0
    avg_sparsity = 0
    avg_l1 = 0
    avg_proximity_loss = 0
    avg_generations = 0
    for instance_id, dictofit in results_baseline.items():
        if instance_id in ["population_generation_time", "prepare_instances_time"]:
            continue
        if dictofit["Single_run"]["Time_intermediate_from_scratch"] is np.inf or dictofit["Single_run"]["Time_intermediate_from_scratch"] == None:
            continue
        time_baseline += dictofit["Single_run"]["Time_intermediate_from_scratch"]
        avg_cfes_found += dictofit["Multiple_runs"]["Times_that_at_least_one_cfe_found_percentage"]
        avg_l2 += dictofit["Single_run"]["Best_cfe_l2_distance"] / ugce_obj.max_l2_distance
        avg_l1 += dictofit["Single_run"]["Best_cfe_l1_distance"] / ugce_obj.max_l1_distance
        avg_proximity_loss += dictofit["Single_run"]["Best_cfe_weighted_l1_distance"]
        avg_sparsity += dictofit["Single_run"]["Best_cfe_sparsity"] / ugce_obj.max_sparsity
        avg_generations += dictofit["Single_run"]["Num_generations"]

    avg_l2 /= avg_cfes_found
    avg_sparsity /= avg_cfes_found
    avg_generations /=  avg_cfes_found
    avg_l1 /= avg_cfes_found
    avg_proximity_loss /= avg_cfes_found

    time_baseline += results_baseline["population_generation_time"]
    time_baseline += results_baseline["prepare_instances_time"]
    if time_baseline > 60:
        time_baseline = time_baseline/60
    #     print("Time taken for generating counterfactuals using UGCE from scratch: ", time_baseline, " minutes")
    # else:
    #     print("Time taken for generating counterfactuals using UGCE from scratch: ", time_baseline)
    # print(f"Avg CFES found: {(avg_cfes_found/len(results_baseline) * 100)}%")
    # print("Avg distance: ", avg_l2)
    # print("Avg l1: ", avg_l1)
    # print("Avg proximity loss: ", avg_proximity_loss)
    # print("Avg sparsity: ", avg_sparsity)
    # print("Avg generations: ", avg_generations)
    return time_baseline, avg_generations, (avg_cfes_found/len(results_baseline) * 100), avg_l2, avg_l1, avg_proximity_loss, avg_sparsity

def aggregate_results_baseline(ugce_object, results_baseline_explainer):
    # Placeholder for collecting results across all runs
    full_times = []
    coverages = []
    distances = []
    l1s = []
    proximities = []
    sparsities = []
    generation_counts = []

    # Run the evaluation loop
    for baseline_explainer in results_baseline_explainer:
        ugce_baseline_ful_time, ugce_baseline_generations, ugce_baseline_coverage, ugce_baseline_avg_distance, ugce_baseline_avg_l1, \
            ugce_baseline_avg_proximity_loss, ugce_baseline_avg_sparsity = stats_baseline(ugce_object, baseline_explainer)
        
        full_times.append(ugce_baseline_ful_time)
        distances.append(ugce_baseline_avg_distance)
        l1s.append(ugce_baseline_avg_l1)
        proximities.append(ugce_baseline_avg_proximity_loss)
        sparsities.append(ugce_baseline_avg_sparsity)
        generation_counts.append(ugce_baseline_generations)
        coverages.append(ugce_baseline_coverage)

    print_metric_stats("Full Time", full_times)
    print_metric_stats("Generations", generation_counts)
    print_metric_stats("Coverage", coverages)
    # print_metric_stats("L2 Distance", distances)
    # print_metric_stats("L1 Distance", l1s)
    print_metric_stats("Proximity Loss", proximities)
    print_metric_stats("Sparsity", sparsities)

################################### UGCE-Incremental Evaluation ###################################
def stats_incremental(ugce_obj, results_incremental, return_matrices=False, verbose=False):
    if return_matrices:
        time_dynamic_arr = []
        l2_arr = []
        l1_arr = []
        proximity_loss_arr = []
        sparsity_arr = []
        generations_arr = []
        best_intermediate_best_dist_arr = []
        cfes_found_arr = []

        time_dynamic = 0
        avg_l2 = 0
        avg_l1 = 0
        avg_cfes_found = 0
        avg_sparsity = 0
        avg_proximity_loss = 0
        avg_best_intermediate_best_dist = 0
        avg_generations = 0

        for instance_id, dictofit in results_incremental.items():
            if instance_id in ["population_generation_time", "prepare_instances_time"]:
                continue
            if dictofit["Single_run"]["Time_dynamic"] is np.inf or dictofit["Single_run"]["Time_dynamic"] == None:
                cfes_found_arr.append(0)
                continue
            time_dynamic_arr.append(dictofit["Single_run"]["Time_dynamic"])
            cfes_found_arr.append(dictofit["Multiple_runs"]["Times_that_at_least_one_cfe_found_percentage"])
            l2_arr.append(dictofit["Single_run"]["Best_cfe_l2_distance"] / ugce_obj.max_l2_distance)
            l1_arr.append(dictofit["Single_run"]["Best_cfe_l1_distance"] / ugce_obj.max_l1_distance)
            proximity_loss_arr.append(dictofit["Single_run"]["Best_cfe_weighted_l1_distance"][0])
            sparsity_arr.append(dictofit["Single_run"]["Best_cfe_sparsity"] / ugce_obj.max_sparsity)
            generations_arr.append(dictofit["Single_run"]["Num_generations"])
            best_intermediate_best_dist_arr.append(dictofit["Single_run"]["Weighted_L1_distance_between_best_and_intermediate_best_cfes"][0])
            cfes_found_arr.append(1)

            time_dynamic += dictofit["Single_run"]["Time_dynamic"]
            avg_cfes_found += dictofit["Multiple_runs"]["Times_that_at_least_one_cfe_found_percentage"]
            avg_l2 += dictofit["Single_run"]["Best_cfe_l2_distance"] / ugce_obj.max_l2_distance
            avg_l1 += dictofit["Single_run"]["Best_cfe_l1_distance"] / ugce_obj.max_l1_distance
            avg_proximity_loss += dictofit["Single_run"]["Best_cfe_weighted_l1_distance"]
            avg_sparsity += dictofit["Single_run"]["Best_cfe_sparsity"] / ugce_obj.max_sparsity
            avg_generations += dictofit["Single_run"]["Num_generations"]
            avg_best_intermediate_best_dist += dictofit["Single_run"]["Weighted_L1_distance_between_best_and_intermediate_best_cfes"]

        avg_l2 /= avg_cfes_found
        avg_sparsity /= avg_cfes_found
        avg_generations /=  avg_cfes_found
        avg_l1 /= avg_cfes_found
        avg_proximity_loss /= avg_cfes_found
        avg_best_intermediate_best_dist /= avg_cfes_found
        if time_dynamic > 60:
            time_dynamic = time_dynamic/60
        avg_cfes_found = (avg_cfes_found/len(results_incremental) * 100)

        return (time_dynamic_arr, generations_arr, cfes_found_arr, proximity_loss_arr, sparsity_arr, best_intermediate_best_dist_arr),\
                (time_dynamic, avg_generations, avg_cfes_found, avg_l2, avg_l1, avg_proximity_loss, avg_sparsity, avg_best_intermediate_best_dist)
    else:
        time_dynamic = 0
        avg_l2 = 0
        avg_cfes_found = 0
        avg_sparsity = 0
        avg_l1 = 0
        avg_proximity_loss = 0
        avg_best_intermediate_best_dist = 0
        avg_generations = 0
        for instance_id, dictofit in results_incremental.items():
            if instance_id in ["population_generation_time", "prepare_instances_time"]:
                continue
            if dictofit["Single_run"]["Time_dynamic"] is np.inf or dictofit["Single_run"]["Time_dynamic"] == None:
                continue
            time_dynamic += dictofit["Single_run"]["Time_dynamic"]
            avg_cfes_found += dictofit["Multiple_runs"]["Times_that_at_least_one_cfe_found_percentage"]
            avg_l2 += dictofit["Single_run"]["Best_cfe_l2_distance"] / ugce_obj.max_l2_distance
            avg_l1 += dictofit["Single_run"]["Best_cfe_l1_distance"] / ugce_obj.max_l1_distance
            avg_proximity_loss += dictofit["Single_run"]["Best_cfe_weighted_l1_distance"]
            avg_sparsity += dictofit["Single_run"]["Best_cfe_sparsity"] / ugce_obj.max_sparsity
            avg_generations += dictofit["Single_run"]["Num_generations"]
            avg_best_intermediate_best_dist += dictofit["Single_run"]["Weighted_L1_distance_between_best_and_intermediate_best_cfes"]

        avg_l2 /= avg_cfes_found
        avg_sparsity /= avg_cfes_found
        avg_generations /=  avg_cfes_found
        avg_l1 /= avg_cfes_found
        avg_proximity_loss /= avg_cfes_found
        avg_best_intermediate_best_dist /= avg_cfes_found
        
        # time_baseline += results_baseline["population_generation_time"]
        # time_baseline += results_baseline["prepare_instances_time"]
        if time_dynamic > 60:
            if verbose:
                print("Time taken for generating counterfactuals using UGCE dynamic: ", time_dynamic/60, " minutes")
            else:
                print("Time taken for generating counterfactuals using UGCE dynamic: ", time_dynamic)
            time_dynamic = time_dynamic/60
        avg_cfes_found = (avg_cfes_found/len(results_incremental) * 100)
        # print(f"Avg CFES found: {avg_cfes_found}%")
        # print("Avg distance: ", avg_l2)
        # print("Avg L1: ", avg_l1)
        # print("Avg proximity loss: ", avg_proximity_loss)
        # print("Avg sparsity: ", avg_sparsity)
        # print("Avg generations: ", avg_generations)
        # print("Avg best intermediate best distance: ", avg_best_intermediate_best_dist)
        return time_dynamic, avg_generations, avg_cfes_found, avg_l2, avg_l1, avg_proximity_loss, avg_sparsity, avg_best_intermediate_best_dist

def aggregate_results_incremental(ugce_object, results_incremental_explainer, verbose=False):
    # Placeholder for collecting results across all runs
    full_times = []
    coverages = []
    distances = []
    l1s = []
    proximities = []
    sparsities = []
    generation_counts = []
    intermediate_best_distances = []

    # Run the evaluation loop
    for incremental_explainer in results_incremental_explainer:
        ugce_baseline_ful_time, ugce_baseline_generations, ugce_baseline_coverage, ugce_baseline_avg_distance, ugce_baseline_avg_l1, \
            ugce_baseline_avg_proximity_loss, ugce_baseline_avg_sparsity, ugce_avg_intermediate_best_distances = stats_incremental(ugce_object, incremental_explainer, verbose=verbose)
        
        full_times.append(ugce_baseline_ful_time)
        distances.append(ugce_baseline_avg_distance)
        l1s.append(ugce_baseline_avg_l1)
        proximities.append(ugce_baseline_avg_proximity_loss)
        sparsities.append(ugce_baseline_avg_sparsity)
        generation_counts.append(ugce_baseline_generations)
        coverages.append(ugce_baseline_coverage)
        intermediate_best_distances.append(ugce_avg_intermediate_best_distances)

    # Compute and print means and stds
    def print_metric_stats(name, values):
        print(f"{name}: mean = {np.mean(values):.2f}, std = {np.std(values):.2f}")

    if verbose:
        print_metric_stats("Full Time", full_times)
        print_metric_stats("Generations", generation_counts)
        print_metric_stats("Coverage", coverages)
        # print_metric_stats("L2 Distance", distances)
        # print_metric_stats("L1 Distance", l1s)
        print_metric_stats("Proximity Loss", proximities)
        print_metric_stats("Sparsity", sparsities)
        print_metric_stats("Intermediate Best Distances", intermediate_best_distances)
    return np.mean(full_times), np.mean(generation_counts), np.mean(coverages), np.mean(proximities), np.mean(sparsities), np.mean(intermediate_best_distances)

def gather_results_sequence_of_type_constraints(dice_obj, results_incremental_imm_ranges_direct_arr, results_incremental_ranges_imm_incr, results_incremental_dir_im_range_arr, verbose=False):
    if verbose:
        print("Imm → Range → Dir")
    time_dynamic_imm_ranges_direct, avg_generations_imm_ranges_direct, avg_cfes_found_imm_ranges_direct, \
        avg_proximity_loss_imm_ranges_direct, avg_sparsity_imm_ranges_direct, avg_intermediate_imm_ranges_incr = aggregate_results_incremental(dice_obj, results_incremental_imm_ranges_direct_arr, verbose=verbose)
    
    if verbose:
        print("\nRange → Imm → Dir")
    time_dynamic_ranges_imm_incr, avg_generations_ranges_imm_incr, avg_cfes_found_ranges_imm_incr, \
        avg_proximity_loss_ranges_imm_incr, avg_sparsity_ranges_imm_incr, avg_intermediate_ranges_imm_incr = aggregate_results_incremental(dice_obj, results_incremental_ranges_imm_incr, verbose=verbose)

    if verbose:
        print("\nDir → Imm → Range")
    time_dynamic_dir_im_range, avg_generations_dir_im_range, avg_cfes_found_dir_im_range, \
        avg_proximity_loss_dir_im_range, avg_sparsity_dir_im_range, avg_intermediate_dir_im_range = aggregate_results_incremental(dice_obj, results_incremental_dir_im_range_arr, verbose=verbose)
    
    return time_dynamic_imm_ranges_direct, avg_generations_imm_ranges_direct, avg_cfes_found_imm_ranges_direct, avg_proximity_loss_imm_ranges_direct, avg_sparsity_imm_ranges_direct, avg_intermediate_imm_ranges_incr, \
        time_dynamic_ranges_imm_incr, avg_generations_ranges_imm_incr, avg_cfes_found_ranges_imm_incr, avg_proximity_loss_ranges_imm_incr, avg_sparsity_ranges_imm_incr, avg_intermediate_ranges_imm_incr, \
        time_dynamic_dir_im_range, avg_generations_dir_im_range, avg_cfes_found_dir_im_range, avg_proximity_loss_dir_im_range, avg_sparsity_dir_im_range, avg_intermediate_dir_im_range

def plot_constraint_analysis(dice_obj, results_incremental_imm_ranges_direct_arr, results_incremental_ranges_imm_incr, results_incremental_dir_im_range_arr, datasetName):
    time_dynamic_imm_ranges_direct, avg_generations_imm_ranges_direct, avg_cfes_found_imm_ranges_direct, avg_proximity_loss_imm_ranges_direct, avg_sparsity_imm_ranges_direct, avg_intermediate_imm_ranges_incr, \
        time_dynamic_ranges_imm_incr, avg_generations_ranges_imm_incr, avg_cfes_found_ranges_imm_incr, avg_proximity_loss_ranges_imm_incr, avg_sparsity_ranges_imm_incr, avg_intermediate_ranges_imm_incr, \
        time_dynamic_dir_im_range, avg_generations_dir_im_range, avg_cfes_found_dir_im_range, avg_proximity_loss_dir_im_range, avg_sparsity_dir_im_range, avg_intermediate_dir_im_range =\
        gather_results_sequence_of_type_constraints(dice_obj, results_incremental_imm_ranges_direct_arr, results_incremental_ranges_imm_incr, results_incremental_dir_im_range_arr)
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })
    colors = sns.color_palette("pastel")

    # Define constraint sequences
    # constraint_orders = [
    #     "Immutability → Range → Directionality",
    #     "Range → Immutability → Directionality",
    #     "Directionality → Immutability → Range"
    # ]
    constraint_orders = [
        "I → R → D",
        "R → I → D",
        "D → I → R"
    ]

    cfe_found = [
        avg_cfes_found_imm_ranges_direct,
        avg_cfes_found_ranges_imm_incr,
        avg_cfes_found_dir_im_range
    ]

    avg_time = [
        time_dynamic_imm_ranges_direct,
        time_dynamic_ranges_imm_incr,
        time_dynamic_dir_im_range
    ]

    # avg_l2 = [
    #     avg_distance_imm_ranges_incr,
    #     avg_distance_ranges_imm_incr,
    #     avg_distance_dir_im_range
    # ]

    # avg_l1 = [
    #     avg_l1_imm_ranges_incr,
    #     avg_l1_ranges_imm_incr,
    #     avg_l1_dir_im_range
    # ]

    avg_weighted_l1 = [
        avg_proximity_loss_imm_ranges_direct,
        avg_proximity_loss_ranges_imm_incr,
        avg_proximity_loss_dir_im_range
    ]

    avg_sparsity = [
        avg_sparsity_imm_ranges_direct,
        avg_sparsity_ranges_imm_incr,
        avg_sparsity_dir_im_range
    ]

    avg_generations = [
        avg_generations_imm_ranges_direct,
        avg_generations_ranges_imm_incr,
        avg_generations_dir_im_range
    ]

    avg_intermediate = [
        avg_intermediate_imm_ranges_incr,
        avg_intermediate_ranges_imm_incr,
        avg_intermediate_dir_im_range
    ]

    # Plot each metric
    metrics = {
        "CFE Found (\%)": cfe_found,
        "Avg. Time (s)": avg_time,
        "Avg. Generations": avg_generations,
        # "Avg. L2 Distance": avg_l2,
        # "Avg. L1 Distance": avg_l1,
        "Avg. Proximity Loss": avg_weighted_l1,
        "Avg. Sparsity": avg_sparsity,
        "Avg. Intermediate": avg_intermediate
    }

    for metric, values in metrics.items():
        fig, ax = plt.subplots(figsize=(4, 2.8))
        # bars = ax.bar(range(len(constraint_orders)), values, width=0.5, color=colors, edgecolor='black', linewidth=0.5)
        x_pos = np.linspace(0, len(constraint_orders) - 1.4, len(constraint_orders))
        bars = ax.bar(x_pos, values, width=0.5, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x_pos)


        # ax.set_title(f"Effect of Constraint Order on {metric}", pad=8, fontsize=11)
        ax.set_ylabel(metric, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)

        ax.set_ylim(0, max(values) * 1.15)

        # Fix the warning by explicitly setting xticks
        ax.set_xticks(range(len(constraint_orders)))
        ax.set_xticklabels(constraint_orders, rotation=15, ha='right', fontsize=14)

        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        # Add value labels above bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.015,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=14)

        plt.tight_layout(pad=0)
        plt.show()
        os.makedirs(f"{UGCE_dir}/results/assess_ugce/{datasetName}", exist_ok=True)
        fig.savefig(
            f"{UGCE_dir}/results/assess_ugce/{datasetName}/effect_of_order_{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'percent')}.pdf",
            dpi=300,
            bbox_inches='tight'
        )