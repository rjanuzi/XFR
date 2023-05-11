# Face Recognition (FR) - DLIB ResNET Approximation with Genetic Algorithm

import json
import pickle
from datetime import datetime
from math import inf
from pathlib import Path
from random import random, shuffle
from time import time

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from scipy import stats

from util._telegram import send_simple_message

# TODO - Configure to use (or not) blank background in reset parts
# RESNET_FACEPARTS_DISTANCES_FILE = Path("fr", "distances_resnet_faceparts.json")
RESNET_FACEPARTS_DISTANCES_FILE = Path("fr", "distances_resnet_faceparts_nb.json")
DISTANCES_FILES_PKL = Path("fr", "distances.pickle")

# TODO When not using blank background, we need to ignore more combinations
# RESNET_COLS_TO_IGNORE = [
#     "resnet_left_ear",
#     "resnet_right_ear",
#     "resnet_ears",
#     "resnet_full_face",
# ]

RESNET_COLS_TO_IGNORE = [
    "resnet_face",
    "resnet_eyes",
    "resnet_eyebrows",
    "resnet_left_ear",
    "resnet_right_ear",
    "resnet_ears",
    "resnet_upper_lip",
    "resnet_mouth",
    "resnet_mouth_and_nose",
    "resnet_eyes_and_eyebrows",
    "resnet_eyes_and_nose",
    "resnet_full_face",
]

DLIB_DISTANCES_FILE = Path("fr", "distances_dlib.json")
DLIB_DATASET_CLUSTERS_FILE = Path("fr", "dlib_clusters.json")

# TODO When not usng blank background, we need to adjust the name of the experiments
# RESULTS_FOLDER = Path("experiments", f"{datetime.now().strftime('%Y%m%d%H%M%S')}")
RESULTS_FOLDER = Path(
    "experiments", f"{datetime.now().strftime('%Y%m%d%H%M%S')}_results_nb"
)
RESULTS_FOLDER.mkdir(exist_ok=True)

# RESULTS_FILE = RESULTS_FOLDER.joinpath("experiments.csv")
RESULTS_FILE = RESULTS_FOLDER.joinpath("experiments_nb.csv")

# Experiments params

# AG Search Params
CXPB = [0.3]  # Probability with which two individuals are crossed
MUTPB = [0.2]  # Probability for mutating an individual
INDPB = [
    0.2,
]  # Probability for flipping a bit of an individual
POP_SIZE = [200, 400, 800]  # Population size
MAX_GENERATIONS = [50, 100, 200, 400, 800, 1000]  # Maximum number of generations

SUB_SET_SIZE = 1000000  # Number of distances to consider
NO_BEST_MAX_GENERATIONS = 20  # Reset pop if no improvement in the last N generations

RANK_ERROR_IMGS_LIMIT = 200
RANK_ERROR_MIN_IMGS = 25

STEP_ERROR_DLIB_THRESHOLD = 0.5

RECOVER_FITNESS_IMGS_LIMIT = 200
RECOVER_FITNESS_TOP_N = 25

FITNESS_CACHING_LIMIT = 1000000


def load_dlib_df_distances() -> pd.DataFrame:
    # Try to load the pre-processed distances from pickle file
    try:
        print("Loading DLIB distances from pickle file...")
        distances = pickle.load(open(DISTANCES_FILES_PKL, "rb"))
    except:
        # Load distances from raw files into dataframes
        # DLIB Distances ( <pair>: {'dlib': distance}} )
        print("No distances pickle file found. Loading DLIB Raw distances data...")
        tmp_raw_data = json.load(open(DLIB_DISTANCES_FILE, "r"))

        dlib_distances = pd.DataFrame(
            dict(
                pair=tmp_raw_data.keys(),
                dlib_distance=(d["dlib"] for d in tmp_raw_data.values()),
            )
        )
        del tmp_raw_data

        # ResNET Faceparts Distances
        def rows_generator(resnet_faceparts_raw_data):
            for pair, distances in resnet_faceparts_raw_data.items():
                distances.update({"pair": pair})
                yield distances

        print("Loading ResNET Faceparts distances from json file...")
        tmp_raw_data = json.load(open(RESNET_FACEPARTS_DISTANCES_FILE, "r"))

        generator = rows_generator(tmp_raw_data)
        del tmp_raw_data

        resnet_faceparts_distances = pd.DataFrame(generator)

        print("ResNET Faceparts raw data loaded")

        # Join distances into a sigle dataframe
        distances = dlib_distances.merge(
            resnet_faceparts_distances, on="pair", how="outer"
        )

        del dlib_distances
        del resnet_faceparts_distances

        print("DLIB and ResNET Faceparts distances joined")

        # Filter only images with "n" (from VGGFACE2)
        distances = distances[distances.pair.apply(lambda p: "n" in p)]

        # Generate extra columns
        distances["img1"] = distances.pair.apply(lambda p: p.split(" x ")[0])
        distances["img2"] = distances.pair.apply(lambda p: p.split(" x ")[1])
        distances["person1"] = distances.img1.apply(lambda p: p.split("_")[0])
        distances["person2"] = distances.img2.apply(lambda p: p.split("_")[0])
        distances["same_person"] = (distances.person1 == distances.person2).apply(
            lambda s: "same" if s else "different"
        )

        # Delete unnecessary columns
        distances.drop(columns="pair", inplace=True)

        print("Distances extra columns generated")

        # Sort columns by name
        distances = distances.reindex(sorted(distances.columns), axis=1)

        # Load clusters
        clusters_ref = pd.DataFrame(
            data=json.load(open(DLIB_DATASET_CLUSTERS_FILE, "r"))
        )
        clusters_ref.set_index("label", inplace=True)

        distances["img1_cluster"] = distances.img1.apply(
            lambda i: clusters_ref.cluster.get(i, None)
        )
        distances["img2_cluster"] = distances.img2.apply(
            lambda i: clusters_ref.cluster.get(i, None)
        )

        del clusters_ref

        print("Clusters data added")

        distances = distances.replace(inf, np.nan)
        distances.dropna(inplace=True)

        distances = distances[
            distances.img1 != distances.img2
        ]  # Remove same image pairs

        print("Saving distances to pickle file...")
        pickle.dump(distances, open(DISTANCES_FILES_PKL, "wb"))

    return distances.round(8).reset_index(drop=True)


# ======================================================================================================
# Run the experiments
# ======================================================================================================

# Caching the individuals for be faster
INDIVIDUALS_FITNESS_CACHE = {}


def individual_to_key(individual):
    return ",".join((str(round(gene, 8)) for gene in individual))


def clear_cached_fitness():
    global INDIVIDUALS_FITNESS_CACHE
    INDIVIDUALS_FITNESS_CACHE = {}


def get_cached_fitness(individual):
    individual_str = individual_to_key(individual=individual)
    return INDIVIDUALS_FITNESS_CACHE.get(individual_str, None)


def add_cached_fitness(individual, fitness):
    global INDIVIDUALS_FITNESS_CACHE
    individual_str = individual_to_key(individual=individual)
    INDIVIDUALS_FITNESS_CACHE[individual_str] = fitness

    if len(INDIVIDUALS_FITNESS_CACHE) > FITNESS_CACHING_LIMIT:
        print("Caching limit. Clearing")
        clear_cached_fitness()


# Fitness Function
def rank_error(individual, cluster_norm_distances, resnet_distances_norm, imgs):
    """
    Calculate the Mean Squared Error (MSE) of the individual as a measure of fitness
    """
    individual_sum = sum(individual)

    if individual_sum == 0:
        return (inf,)

    individual = [i / individual_sum for i in individual]

    cached_fitness = get_cached_fitness(individual)
    if cached_fitness is not None:
        return cached_fitness

    # Remove equal images
    norm_distances = cluster_norm_distances[
        cluster_norm_distances.img1 != cluster_norm_distances.img2
    ].copy()

    # Calculate the Distance with the ResNet Combination
    norm_distances.loc[:, "combination"] = resnet_distances_norm.dot(individual)
    norm_distances.combination = norm_distances.combination.round(8)

    # Sort by the Dlib Distance and the Combination Distance.
    # The distances are sorted by dlib by default
    by_comb_distances = norm_distances.sort_values(
        by="combination", ascending=True, ignore_index=True
    )

    corrs = []
    for img in imgs[:RANK_ERROR_IMGS_LIMIT]:
        dlib_img2_sequence = (
            norm_distances[norm_distances.img1 == img].reset_index(drop=True).img2
        )

        if len(dlib_img2_sequence) < RANK_ERROR_MIN_IMGS:
            continue

        comb_img2_sequence = (
            by_comb_distances[by_comb_distances.img1 == img].reset_index(drop=True).img2
        )

        tmp_corr = dlib_img2_sequence.corr(comb_img2_sequence, method="kendall")

        if not np.isnan(tmp_corr):
            corrs.append(round(tmp_corr, 8))

    # The Search algorithm will try to minimize the error and we need to maximize the correlation
    fitness = (round(np.mean(corrs), 8) * -1,)

    add_cached_fitness(individual=individual, fitness=fitness)

    return fitness


def recover_fitness(individual, cluster_norm_distances, resnet_distances_norm, imgs):
    individual_sum = sum(individual)

    if individual_sum == 0:
        return (inf,)

    individual = [i / individual_sum for i in individual]

    cached_fitness = get_cached_fitness(individual)
    if cached_fitness is not None:
        return cached_fitness

    # Remove equal images
    norm_distances = cluster_norm_distances[
        cluster_norm_distances.img1 != cluster_norm_distances.img2
    ].copy()

    # Calculate the Distance with the ResNet Combination
    norm_distances.loc[:, "combination"] = resnet_distances_norm.dot(individual)
    norm_distances.combination = norm_distances.combination.round(8)

    # Sort by the Dlib Distance and the Combination Distance.
    # The distances are sorted by dlib by default
    by_comb_distances = norm_distances.sort_values(
        by="combination", ascending=True, ignore_index=True
    )

    top_n_count = []
    for img in imgs[:RECOVER_FITNESS_IMGS_LIMIT]:
        dlib_img2_sequence = (
            norm_distances[norm_distances.img1 == img]
            .reset_index(drop=True)
            .img2[:RECOVER_FITNESS_TOP_N]
        )

        if len(dlib_img2_sequence) < RECOVER_FITNESS_TOP_N:
            continue

        comb_img2_sequence = (
            by_comb_distances[by_comb_distances.img1 == img]
            .reset_index(drop=True)
            .img2[:RECOVER_FITNESS_TOP_N]
        )

        top_n_count.append(
            len(comb_img2_sequence[comb_img2_sequence.isin(dlib_img2_sequence)])
        )

    # The Search algorithm will try to minimize the error and we need to maximize the correlation
    fitness = (round(np.mean(top_n_count), 8) * -1,)

    add_cached_fitness(individual=individual, fitness=fitness)

    return fitness


def mse(individual, cluster_norm_distances, resnet_distances_norm, imgs):
    """
    Calculate the Mean Squared Error (MSE) of the individual as a measure of fitness
    """
    individual_sum = sum(individual)

    if individual_sum == 0:
        return (inf,)

    individual = [i / individual_sum for i in individual]

    cached_fitness = get_cached_fitness(individual)
    if cached_fitness is not None:
        return cached_fitness

    cluster_norm_distances = cluster_norm_distances[
        cluster_norm_distances.img1 != cluster_norm_distances.img2
    ].copy()

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.dot(individual)
    cluster_norm_distances.loc[:, "error"] = (
        cluster_norm_distances.combination - cluster_norm_distances.dlib_distance
    )
    cluster_norm_distances.loc[:, "sqr_error"] = (
        cluster_norm_distances.error.abs() + 1
    ) ** 2  # Avoid squared of fractions

    # Shall return a tuple for compatibility with DEAP
    fitness = (
        cluster_norm_distances[
            cluster_norm_distances.sqr_error != inf
        ].sqr_error.mean(),
    )

    add_cached_fitness(individual=individual, fitness=fitness)

    return fitness


def mae(individual, cluster_norm_distances, resnet_distances_norm, imgs):
    """
    Calculate the Mean Absolute Error (MAE) of the individual as a measure of fitness
    """

    individual_sum = sum(individual)

    if individual_sum == 0:
        return (inf,)

    individual = [i / individual_sum for i in individual]

    cached_fitness = get_cached_fitness(individual)
    if cached_fitness is not None:
        return cached_fitness

    cluster_norm_distances = cluster_norm_distances[
        cluster_norm_distances.img1 != cluster_norm_distances.img2
    ].copy()

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.dot(individual)
    cluster_norm_distances.loc[:, "error"] = (
        cluster_norm_distances.combination - cluster_norm_distances.dlib_distance
    )

    # Shall return a tuple for compatibility with DEAP
    fitness = (
        cluster_norm_distances[cluster_norm_distances.error != inf].error.abs().mean(),
    )

    add_cached_fitness(individual=individual, fitness=fitness)

    return fitness


def abs_error(individual, cluster_norm_distances, resnet_distances_norm, imgs):
    """
    Calculate the Absolute Error Sum of the individual as a measure of fitness
    """

    individual_sum = sum(individual)

    if individual_sum == 0:
        return (inf,)

    individual = [i / individual_sum for i in individual]

    cached_fitness = get_cached_fitness(individual)
    if cached_fitness is not None:
        return cached_fitness

    cluster_norm_distances = cluster_norm_distances[
        cluster_norm_distances.img1 != cluster_norm_distances.img2
    ].copy()

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.dot(individual)
    cluster_norm_distances.loc[:, "error"] = (
        cluster_norm_distances.combination - cluster_norm_distances.dlib_distance
    )

    # Shall return a tuple for compatibility with DEAP
    fitness = (
        cluster_norm_distances[cluster_norm_distances.error != inf].error.abs().sum(),
    )

    add_cached_fitness(individual=individual, fitness=fitness)

    return fitness


def mape_error(individual, cluster_norm_distances, resnet_distances_norm, imgs):
    """
    Calculate the Absolute Error Sum of the individual as a measure of fitness
    """

    individual_sum = sum(individual)

    if individual_sum == 0:
        return (inf,)

    individual = [i / individual_sum for i in individual]

    cached_fitness = get_cached_fitness(individual)
    if cached_fitness is not None:
        return cached_fitness

    cluster_norm_distances = cluster_norm_distances[
        cluster_norm_distances.img1 != cluster_norm_distances.img2
    ].copy()

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.dot(individual)
    cluster_norm_distances.loc[:, "error"] = (
        np.fabs(
            cluster_norm_distances.dlib_distance - cluster_norm_distances.combination
        )
        / cluster_norm_distances.dlib_distance
    )

    # Shall return a tuple for compatibility with DEAP
    fitness = (
        cluster_norm_distances[cluster_norm_distances.error != inf].error.mean(),
    )

    add_cached_fitness(individual=individual, fitness=fitness)

    return fitness


def step_error(individual, cluster_norm_distances, resnet_distances_norm, imgs):
    """
    Calculate the Step differente of the individual as a measure of fitness
    """

    individual_sum = sum(individual)

    if individual_sum == 0:
        return (inf,)

    individual = [(i / individual_sum) for i in individual]

    cached_fitness = get_cached_fitness(individual)
    if cached_fitness is not None:
        return cached_fitness

    cluster_norm_distances = cluster_norm_distances[
        cluster_norm_distances.img1 != cluster_norm_distances.img2
    ].copy()

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.dot(individual)

    # Pandas Like Error
    cluster_norm_distances.loc[
        :, "dlib_same_person"
    ] = cluster_norm_distances.dlib_distance.apply(
        lambda c: 1 if c < STEP_ERROR_DLIB_THRESHOLD else 0
    )
    cluster_norm_distances.loc[
        :, "comb_same_person"
    ] = cluster_norm_distances.combination.apply(
        lambda c: 1 if c < STEP_ERROR_DLIB_THRESHOLD else 0
    )
    cluster_norm_distances.loc[:, "error"] = (
        cluster_norm_distances.comb_same_person
        - cluster_norm_distances.dlib_same_person
    )

    # Shall return a tuple for compatibility with DEAP
    fitness = (
        cluster_norm_distances[cluster_norm_distances.error != inf].error.abs().sum(),
    )

    add_cached_fitness(individual=individual, fitness=fitness)

    return fitness


ERROR_FUNCTIONS = {
    "mse": mse,
    "mae": mae,
    "abs_error": abs_error,
    "mape": mape_error,
    "step_error": step_error,
    "rank_error": rank_error,
    "recover_fitness": recover_fitness,
}
ERROR_FUNCTIONS_NAMES = list(ERROR_FUNCTIONS.keys())


def calc_rank(
    individual,
    cluster_norm_distances,
    resnet_distances_norm,
    save_data=False,
    files_prepend="",
    output_files_folders=None,
    use_scipy=False,
):
    individual_sum = sum(individual)
    individual = [i / individual_sum for i in individual]

    # Remove equal images
    norm_distances = cluster_norm_distances[
        cluster_norm_distances.img1 != cluster_norm_distances.img2
    ].copy()

    # Calculate the Distance with the ResNet Combination
    norm_distances.loc[:, "combination"] = resnet_distances_norm.dot(individual)

    # Sort by the Dlib Distance and the Combination Distance
    by_dlib_distances = norm_distances.sort_values(
        by="dlib_distance", ascending=True, ignore_index=True
    )
    by_comb_distances = norm_distances.sort_values(
        by="combination", ascending=True, ignore_index=True
    )

    if save_data:
        if output_files_folders is None:
            by_dlib_distances.to_excel(files_prepend + "distances.xlsx")
        else:
            output_path = output_files_folders.joinpath(
                files_prepend + "distances.xlsx"
            )
            by_dlib_distances.to_excel(output_path)

    imgs = by_dlib_distances.img1.unique()
    corrs = []
    corrs_by_img = []

    for img in imgs:
        dlib_img2_sequence = (
            by_dlib_distances[by_dlib_distances.img1 == img].reset_index(drop=True).img2
        )

        if len(dlib_img2_sequence) < RANK_ERROR_MIN_IMGS:
            continue

        comb_img2_sequence = (
            by_comb_distances[by_comb_distances.img1 == img].reset_index(drop=True).img2
        )

        if use_scipy:
            tmp_corr = stats.kendalltau(
                x=dlib_img2_sequence, y=comb_img2_sequence.tolist()
            ).correlation
        else:
            tmp_corr = dlib_img2_sequence.corr(comb_img2_sequence, method="kendall")

        if not np.isnan(tmp_corr):
            corrs.append(tmp_corr)

        if save_data:
            corrs_by_img.append(
                {
                    "img": img,
                    "corr": tmp_corr,
                }
            )

    if save_data:
        if output_files_folders is None:
            pd.DataFrame(corrs_by_img).to_excel(files_prepend + "corrs.xlsx")
        else:
            output_path = output_files_folders.joinpath(files_prepend + "corrs.xlsx")
            pd.DataFrame(corrs_by_img).to_excel(output_path)

    return (
        round(np.min(corrs), 8),
        round(np.max(corrs), 8),
        round(np.median(corrs), 8),
        round(np.mean(corrs), 8),
    )


def run_experiment(params_comb=None):
    distances = load_dlib_df_distances()
    clusters = set(distances.img1_cluster.unique()).union(
        set(distances.img2_cluster.unique())
    )
    clusters = sorted(clusters)

    print("Distances data loaded")

    # Individuals representation
    resnet_cols = list(
        filter(
            lambda c: ("resnet" in c) and (c not in RESNET_COLS_TO_IGNORE),
            distances.columns,
        )
    )

    IND_SIZE = len(resnet_cols)

    with open(RESULTS_FILE, "w") as f:
        f.write(
            "exp_id,cluster,error_function,total_pairs,total_persons,cxpb,mtpb,indpb,pop_size,max_generations,no_best_max_gens,best_generation,best_fitness,min_rank,max_rank,median_rank,mean_rank,exec_time_sec\n"
        )

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Error (minimize)
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # If no params is provided, use the available one by default
    if params_comb is None:

        def params_generator():
            for max_generations in MAX_GENERATIONS:
                for pop_size in POP_SIZE:
                    for indpb in INDPB:
                        for mutpb in MUTPB:
                            for cxpb in CXPB:
                                for error_fun_name in ERROR_FUNCTIONS_NAMES:
                                    yield {
                                        "cxpb": cxpb,
                                        "mutpb": mutpb,
                                        "indpb": indpb,
                                        "pop_size": pop_size,
                                        "max_generations": max_generations,
                                        "error_fun": ERROR_FUNCTIONS[error_fun_name],
                                    }

        params_comb = list(params_generator())
    else:
        params_comb = list(
            map(
                lambda p: {**p, "error_fun": ERROR_FUNCTIONS[p["error_fun"]]},
                params_comb,
            )
        )

    send_simple_message(
        f"Starting DLIB ResNET GA Experiments with {len(params_comb)} combination of parameters"
    )

    params_experimented = 0
    exp_id = 0
    for params in params_comb:
        params_start_time = time()
        params_experimented += 1
        current_cxpb = params["cxpb"]
        current_mutpb = params["mutpb"]
        current_indpb = params["indpb"]
        current_pop_size = params["pop_size"]
        current_max_generations = params["max_generations"]
        current_error_fun = params["error_fun"]

        best = {}
        for cluster in clusters:
            clear_cached_fitness()
            exp_id += 1
            cluster_distances = distances[
                (distances.img1_cluster == cluster)
                & (distances.img2_cluster == cluster)
            ]

            cluster_distances = cluster_distances.iloc[:SUB_SET_SIZE].reset_index(
                drop=True
            )

            total_pairs = len(cluster_distances)
            total_persons = cluster_distances.person1.unique().shape[0]
            print(
                f"""
                    Experiment {exp_id} with {total_pairs} pairs of images of {total_persons} persons
                    Cluster: {cluster}
                    Error Function: {current_error_fun.__name__}
                    CXPB: {current_cxpb}
                    MUTPB: {current_mutpb}
                    INDPB: {current_indpb}
                    POP_SIZE: {current_pop_size}
                    MAX_GENERATIONS: {current_max_generations}
                    """
            )

            # Normalize distances inside cluster
            cluster_norm_distances = cluster_distances.copy()
            cluster_norm_distances = cluster_norm_distances.round(8)

            # Normalize numerical columns
            for col in resnet_cols + ["dlib_distance"]:
                cluster_norm_distances[col] = (
                    cluster_norm_distances[col] - cluster_norm_distances[col].min()
                ) / (
                    cluster_norm_distances[col].max()
                    - cluster_norm_distances[col].min()
                )

            cluster_norm_distances = cluster_norm_distances.round(8)
            cluster_norm_distances = cluster_norm_distances.sort_values(
                by="dlib_distance", ascending=True, ignore_index=True
            )

            resnet_distances_norm = cluster_norm_distances.loc[:, resnet_cols]

            # imgs list to be used at rank_error function
            imgs = list(cluster_norm_distances.img1.unique())
            shuffle(imgs)

            # Prepare DEAP
            toolbox = base.Toolbox()
            toolbox.register("attr_float", random)
            toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                toolbox.attr_float,
                n=IND_SIZE,
            )
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register(
                "evaluate",
                current_error_fun,
                cluster_norm_distances=cluster_norm_distances,
                resnet_distances_norm=resnet_distances_norm,
                imgs=imgs,
            )
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=current_indpb)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Start AG Search
            start_time = time()
            pop = toolbox.population(n=current_pop_size)

            fitness_time = time()
            print(f"Evaluating {current_pop_size} individuals")
            # Evaluate the entire population
            for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
                ind.fitness.values = fit
            print(f"Time to evaluate fitness {(time() - fitness_time)//60} minutes")

            # Extracting all the fitnesses of
            fits = [ind.fitness.values[0] for ind in pop]

            # Variable keeping track of the number of generations
            g = 0

            # Output files for best individuals
            individuals_folder = RESULTS_FOLDER.joinpath(
                f"{str(exp_id).zfill(5)}_individuals"
            )
            individuals_folder.mkdir(exist_ok=True)
            best_individual_file = individuals_folder.joinpath("best_individual.json")
            best_individuals_file = individuals_folder.joinpath("best_individuals.json")

            low_std_times = 0
            # last_max_fit = -1e9
            last_min_fit = +1e9
            best_generation = 0
            last_gen_reset = 0
            bests = []

            # Begin the evolution
            while g < current_max_generations:
                # A new generation
                g = g + 1
                print(f"Generation {g}")

                # Select the next generation individuals
                offspring = toolbox.select(pop, len(pop))

                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))

                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random() < current_cxpb:
                        toolbox.mate(child1, child2)
                        # Invalidate fitnesses for the new individuals
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random() < current_mutpb:
                        toolbox.mutate(mutant)
                        # Invalidate fitnesses for the new individual
                        del mutant.fitness.values

                # Evaluate the individuals with an invalid fitness (The new ones)
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Replace population with offspring
                pop[:] = offspring

                # Gather all the fitnesses in one list and print the stats
                fits = [ind.fitness.values[0] for ind in pop]

                length = len(pop)
                mean = sum(fits) / length
                sum2 = sum(x * x for x in fits)
                std = abs(sum2 / length - mean**2) ** 0.5

                if std < 3e-3:
                    low_std_times += 1

                # If we face NO_BEST_MAX_GENERATIONS generations without improvement, reset the population
                if (g - best_generation > NO_BEST_MAX_GENERATIONS) and (
                    g - last_gen_reset > NO_BEST_MAX_GENERATIONS
                ):
                    last_gen_reset = g
                    low_std_times = 0

                    # Reset Pop
                    print("No best found for too long, Resetting pop")
                    pop = toolbox.population(n=current_pop_size)

                    # Evaluate the entire population
                    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
                        ind.fitness.values = fit

                    # Extracting all the fitnesses of
                    fits = [ind.fitness.values[0] for ind in pop]

                # Minimization (Error)
                if min(fits) < last_min_fit:
                    last_min_fit = min(fits)
                    best_generation = g
                    best_idx = fits.index(last_min_fit)
                    best = pop[best_idx]

                    print(f"New best found (Gen: {best_generation}): {last_min_fit}")

                    bests.append(
                        {
                            "generation": best_generation,
                            "fitness": last_min_fit,
                            "best_data": dict(zip(resnet_cols, best)),
                        }
                    )

            # Save results to files
            json.dump(dict(zip(resnet_cols, best)), open(best_individual_file, "w"))
            json.dump(bests, open(best_individuals_file, "w"))
            min_rank, max_rank, median_rank, mean_rank = calc_rank(
                best,
                cluster_norm_distances,
                resnet_distances_norm,
                save_data=False,
                output_files_folders=individuals_folder,
                use_scipy=False,
            )

            with open(RESULTS_FILE, "a") as f:
                tmp_line = f"{exp_id},{cluster},{current_error_fun.__name__},{total_pairs},{total_persons},{current_cxpb},{current_mutpb}"
                tmp_line += f",{current_indpb},{current_pop_size},{current_max_generations},{NO_BEST_MAX_GENERATIONS},{best_generation},{last_min_fit}"
                tmp_line += f",{min_rank},{max_rank},{median_rank},{mean_rank},{int(time()-start_time)}\n"
                f.write(tmp_line)

        if params_experimented % 10 == 0:
            print(
                f"DLIB ResNET GA Experiments:  {params_experimented}/{len(params_comb)} {round(100*params_experimented/len(params_comb),2)}% | Spent {round((time()-params_start_time)//60,2)} min"
            )
            _ = send_simple_message(
                f"DLIB ResNET GA Experiments:  {params_experimented}/{len(params_comb)} {round(100*params_experimented/len(params_comb),2)}% | Spent {round((time()-params_start_time)//60,2)} min"
            )


def run_experiment_v2(params_comb=None, verbose=False):
    distances = load_dlib_df_distances()
    clusters = set(distances.img1_cluster.unique()).union(
        set(distances.img2_cluster.unique())
    )
    clusters = sorted(clusters)

    print("Distances data loaded")

    # Individuals representation
    resnet_cols = list(
        filter(
            lambda c: ("resnet" in c) and (c not in RESNET_COLS_TO_IGNORE),
            distances.columns,
        )
    )

    IND_SIZE = len(resnet_cols)

    with open(RESULTS_FILE, "w") as f:
        f.write(
            "exp_id,cluster,error_function,total_pairs,total_persons,cxpb,mtpb,indpb,pop_size,max_generations,best_generation,best_fitness,min_rank,max_rank,median_rank,mean_rank,exec_time_sec\n"
        )

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Error (minimize)
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # If no params is provided, use the available one by default
    if params_comb is None:

        def params_generator():
            for max_generations in MAX_GENERATIONS:
                for pop_size in POP_SIZE:
                    for indpb in INDPB:
                        for mutpb in MUTPB:
                            for cxpb in CXPB:
                                for error_fun_name in ERROR_FUNCTIONS_NAMES:
                                    yield {
                                        "cxpb": cxpb,
                                        "mutpb": mutpb,
                                        "indpb": indpb,
                                        "pop_size": pop_size,
                                        "max_generations": max_generations,
                                        "error_fun": ERROR_FUNCTIONS[error_fun_name],
                                    }

        params_comb = list(params_generator())
    else:
        params_comb = list(
            map(
                lambda p: {**p, "error_fun": ERROR_FUNCTIONS[p["error_fun"]]},
                params_comb,
            )
        )

    send_simple_message(
        f"Starting DLIB ResNET GA Experiments with {len(params_comb)} combination of parameters"
    )

    params_experimented = 0
    exp_id = 0
    for params in params_comb:
        params_start_time = time()
        params_experimented += 1
        current_cxpb = params["cxpb"]
        current_mutpb = params["mutpb"]
        current_indpb = params["indpb"]
        current_pop_size = params["pop_size"]
        current_max_generations = params["max_generations"]
        current_error_fun = params["error_fun"]

        best = {}
        for cluster in clusters:
            clear_cached_fitness()
            exp_id += 1
            cluster_distances = distances[
                (distances.img1_cluster == cluster)
                & (distances.img2_cluster == cluster)
            ]

            cluster_distances = cluster_distances.iloc[:SUB_SET_SIZE].reset_index(
                drop=True
            )

            total_pairs = len(cluster_distances)
            total_persons = cluster_distances.person1.unique().shape[0]
            print(
                f"""
                    Experiment {exp_id} with {total_pairs} pairs of images of {total_persons} persons
                    Cluster: {cluster}
                    Error Function: {current_error_fun.__name__}
                    CXPB: {current_cxpb}
                    MUTPB: {current_mutpb}
                    INDPB: {current_indpb}
                    POP_SIZE: {current_pop_size}
                    MAX_GENERATIONS: {current_max_generations}
                    """
            )

            # Normalize distances inside cluster
            cluster_norm_distances = cluster_distances.copy()
            cluster_norm_distances = cluster_norm_distances.round(6)

            # Normalize numerical columns
            for col in resnet_cols + ["dlib_distance"]:
                cluster_norm_distances[col] = (
                    cluster_norm_distances[col] - cluster_norm_distances[col].min()
                ) / (
                    cluster_norm_distances[col].max()
                    - cluster_norm_distances[col].min()
                )

            cluster_norm_distances = cluster_norm_distances.round(6)
            cluster_norm_distances = cluster_norm_distances.sort_values(
                by="dlib_distance", ascending=True, ignore_index=True
            )

            resnet_distances_norm = cluster_norm_distances.loc[:, resnet_cols]

            # imgs list to be used at rank_error function
            imgs = list(cluster_norm_distances.img1.unique())
            shuffle(imgs)

            # Prepare DEAP
            toolbox = base.Toolbox()
            toolbox.register("attr_float", random)
            toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                toolbox.attr_float,
                n=IND_SIZE,
            )
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register(
                "evaluate",
                current_error_fun,
                cluster_norm_distances=cluster_norm_distances,
                resnet_distances_norm=resnet_distances_norm,
                imgs=imgs,
            )

            toolbox.register("mate", tools.cxSimulatedBinary, eta=0.3)
            toolbox.register("mutate", tools.mutFlipBit, indpb=current_indpb)
            toolbox.register("select", tools.selTournament, tournsize=3)

            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            mstats = tools.MultiStatistics(fitness=stats_fit)
            mstats.register("max", np.max)
            mstats.register("min", np.min)
            mstats.register("mean", np.mean)
            mstats.register("median", np.median)
            mstats.register("stddev", np.std)

            hof = tools.HallOfFame(maxsize=20)

            # Start AG Search
            start_time = time()
            pop = toolbox.population(n=current_pop_size)
            final_pop, logbook = algorithms.eaSimple(
                population=pop,
                toolbox=toolbox,
                cxpb=current_cxpb,
                mutpb=current_mutpb,
                ngen=current_max_generations,
                stats=mstats,
                halloffame=hof,
                verbose=verbose,
            )

            # Output files for best individuals
            individuals_folder = RESULTS_FOLDER.joinpath(
                f"{str(exp_id).zfill(5)}_individuals"
            )
            individuals_folder.mkdir(exist_ok=True)
            best_individual_file = individuals_folder.joinpath("best_individual.json")
            best_individuals_file = individuals_folder.joinpath("best_individuals.json")
            log_file = individuals_folder.joinpath("evolution_log.json")

            # Save results to files
            best = hof[0]
            bests = [
                {
                    "generation": 0,
                    "fitness": 0,
                    "best_data": dict(zip(resnet_cols, ind)),
                }
                for ind in hof
            ]

            json.dump(dict(zip(resnet_cols, best)), open(best_individual_file, "w"))
            json.dump(bests, open(best_individuals_file, "w"))
            min_rank, max_rank, median_rank, mean_rank = calc_rank(
                best,
                cluster_norm_distances,
                resnet_distances_norm,
                save_data=False,
                output_files_folders=individuals_folder,
                use_scipy=False,
            )

            log = logbook.chapters["fitness"]
            json.dump(log, open(log_file, "w"))

            with open(RESULTS_FILE, "a") as f:
                tmp_line = f"{exp_id},{cluster},{current_error_fun.__name__},{total_pairs},{total_persons},{current_cxpb},{current_mutpb}"
                tmp_line += f",{current_indpb},{current_pop_size},{current_max_generations},'na','na'"
                tmp_line += f",{min_rank},{max_rank},{median_rank},{mean_rank},{int(time()-start_time)}\n"
                f.write(tmp_line)

        if params_experimented % 10 == 0:
            print(
                f"DLIB ResNET GA Experiments:  {params_experimented}/{len(params_comb)} {round(100*params_experimented/len(params_comb),2)}% | Spent {round((time()-params_start_time)//60,2)} min"
            )
            _ = send_simple_message(
                f"DLIB ResNET GA Experiments:  {params_experimented}/{len(params_comb)} {round(100*params_experimented/len(params_comb),2)}% | Spent {round((time()-params_start_time)//60,2)} min"
            )
