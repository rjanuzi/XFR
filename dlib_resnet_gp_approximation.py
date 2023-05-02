# Face Recognition (FR) - DLIB ResNET Approximation with Genetic Algorithm

import json
import math
import operator
from datetime import datetime
from math import inf
from pathlib import Path
from random import randint, seed, shuffle
from time import time

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools

from util._telegram import send_simple_message

# TODO - Configure to use (or not) blank background in reset parts
# RESNET_FACEPARTS_DISTANCES_FILE = Path("fr", "distances_resnet_faceparts.json")
RESNET_FACEPARTS_DISTANCES_FILE = Path("fr", "distances_resnet_faceparts_nb.json")

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

RESULTS_FOLDER = Path(
    "experiments", f"{datetime.now().strftime('%Y%m%d%H%M%S')}_results_gp_nb"
)
RESULTS_FOLDER.mkdir(exist_ok=True)

RESULTS_FILE = RESULTS_FOLDER.joinpath("experiments_nb.csv")

# Experiments params

# AG Search Params
CXPB = [0.3, 0.4, 0.5]  # Probability with which two individuals are crossed
MUTPB = [0.2, 0.3]  # Probability for mutating an individual
INDPB = [
    0.15,
    0.2,
    0.25,
    0.3,
]  # Probability for flipping a bit of an individual
POP_SIZE = [100, 200, 400]  # Population size
MAX_GENERATIONS = [50, 100, 200, 400, 800]  # Maximum number of generations

SUB_SET_SIZE = 1000000  # Number of distances to consider
NO_BEST_MAX_GENERATIONS = 20  # Reset pop if no improvement in the last N generations
HALL_OF_FAME_SIZE = 20  # Number of best individuals to keep in the hall of fame

RANK_ERROR_IMGS_LIMIT = 15

RAND_SEED = 318


def load_dlib_df_distances() -> pd.DataFrame:
    print("Loading DLIB distances...")
    # Load distances from raw files into dataframes

    # DLIB Distances ( <pair>: {'dlib': distance}} )
    tmp_raw_data = json.load(open(DLIB_DISTANCES_FILE, "r"))
    dlib_distances = pd.DataFrame(
        dict(
            pair=tmp_raw_data.keys(),
            dlib_distance=(d["dlib"] for d in tmp_raw_data.values()),
        )
    )
    del tmp_raw_data

    print("DLIB raw data loaded")

    # ResNET Faceparts Distances
    def rows_generator(resnet_faceparts_raw_data):
        for pair, distances in resnet_faceparts_raw_data.items():
            distances.update({"pair": pair})
            yield distances

    tmp_raw_data = json.load(open(RESNET_FACEPARTS_DISTANCES_FILE, "r"))

    generator = rows_generator(tmp_raw_data)
    del tmp_raw_data

    resnet_faceparts_distances = pd.DataFrame(generator)

    print("ResNET Faceparts raw data loaded")

    # Join distances into a sigle dataframe
    distances = dlib_distances.merge(resnet_faceparts_distances, on="pair", how="outer")

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
    clusters_ref = pd.DataFrame(data=json.load(open(DLIB_DATASET_CLUSTERS_FILE, "r")))
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

    distances = distances[distances.img1 != distances.img2]  # Remove same image pairs

    return distances.sort_values(by="dlib_distance", ascending=True)


# ======================================================================================================
# Run the experiments
# ======================================================================================================

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
DEAP_ARGS_FROM_TO = dict(
    zip([f"ARG{i}" for i in range(1, len(resnet_cols))], resnet_cols)
)

# Fitness Function
def rank_error_gp(individual, cluster_norm_distances, resnet_distances_norm, toolbox):
    """
    Calculate the Mean Squared Error (MSE) of the individual as a measure of fitness
    """
    func = toolbox.compile(expr=individual)

    def apply_func(row):
        return func(*row)

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.apply(
        apply_func, axis=1, raw=True
    )

    cluster_norm_distances = cluster_norm_distances.sort_values(
        by="dlib_distance", ascending=True, ignore_index=True
    )
    by_comb_distances = cluster_norm_distances.sort_values(
        by="combination", ascending=True, ignore_index=True
    )

    imgs = list(cluster_norm_distances.img1.unique())
    shuffle(imgs)
    corrs = []
    for img in imgs[:RANK_ERROR_IMGS_LIMIT]:
        dlib_img_distances = cluster_norm_distances[
            cluster_norm_distances.img1 == img
        ].reset_index(drop=True)
        comb_img_distances = by_comb_distances[
            by_comb_distances.img1 == img
        ].reset_index(drop=True)

        tmp_corr = dlib_img_distances.img2.corr(
            comb_img_distances.img2, method="kendall"
        )

        if not np.isnan(tmp_corr):
            corrs.append(tmp_corr)

    return (
        np.mean(corrs) * -1,
    )  # The Search algorithm will try to minimize the error and we need to maximize the correlation


def mse_gp(individual, cluster_norm_distances, resnet_distances_norm, toolbox):
    """
    Calculate the Mean Squared Error (MSE) of the individual as a measure of fitness
    """
    func = toolbox.compile(expr=individual)

    def apply_func(row):
        return func(*row)

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.apply(
        apply_func, axis=1, raw=True
    )
    cluster_norm_distances.loc[:, "error"] = (
        cluster_norm_distances.combination - cluster_norm_distances.dlib_distance
    )

    cluster_norm_distances.loc[:, "sqr_error"] = (
        cluster_norm_distances.error.abs() + 1
    ) ** 2  # Avoid squared of fractions

    return (
        cluster_norm_distances[
            cluster_norm_distances.sqr_error != inf
        ].sqr_error.mean(),
    )  # Shall return a tuple for compatibility with DEAP


def mae_gp(individual, cluster_norm_distances, resnet_distances_norm, toolbox):
    """
    Calculate the Mean Absolute Error (MAE) of the individual as a measure of fitness
    """
    func = toolbox.compile(expr=individual)

    def apply_func(row):
        return func(*row)

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.apply(
        apply_func, axis=1, raw=True
    )
    cluster_norm_distances.loc[:, "error"] = (
        cluster_norm_distances.combination - cluster_norm_distances.dlib_distance
    )

    return (
        cluster_norm_distances[cluster_norm_distances.error != inf].error.abs().mean(),
    )  # Shall return a tuple for compatibility with DEAP


def abs_error_gp(individual, cluster_norm_distances, resnet_distances_norm, toolbox):
    """
    Calculate the Absolute Error Sum of the individual as a measure of fitness
    """
    func = toolbox.compile(expr=individual)

    def apply_func(row):
        return func(*row)

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.apply(
        apply_func, axis=1, raw=True
    )
    cluster_norm_distances.loc[:, "error"] = (
        cluster_norm_distances.combination - cluster_norm_distances.dlib_distance
    )

    return (
        cluster_norm_distances[cluster_norm_distances.error != inf].error.abs().sum(),
    )  # Shall return a tuple for compatibility with DEAP


def step_error_gp(individual, cluster_norm_distances, resnet_distances_norm, toolbox):
    """
    Calculate the Step differente of the individual as a measure of fitness
    """

    func = toolbox.compile(expr=individual)

    def apply_func(row):
        return func(*row)

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.apply(
        apply_func, axis=1, raw=True
    )

    # Pandas Like Error
    cluster_norm_distances.loc[
        :, "dlib_same_person"
    ] = cluster_norm_distances.dlib_distance.apply(lambda c: 1 if c < 0.37 else 0)
    cluster_norm_distances.loc[
        :, "comb_same_person"
    ] = cluster_norm_distances.combination.apply(lambda c: 1 if c < 0.37 else 0)
    cluster_norm_distances.loc[:, "error"] = (
        cluster_norm_distances.comb_same_person
        - cluster_norm_distances.dlib_same_person
    )

    return (
        cluster_norm_distances[cluster_norm_distances.error != inf].error.abs().sum(),
    )  # Shall return a tuple for compatibility with DEAP


ERROR_FUNCTIONS = {
    "mse_gp": mse_gp,
    "mae_gp": mae_gp,
    "abs_error_gp": abs_error_gp,
    "step_error_gp": step_error_gp,
    "rank_error_gp": rank_error_gp,
}
ERROR_FUNCTIONS_NAMES = list(ERROR_FUNCTIONS.keys())


def calc_rank(individual, cluster_norm_distances, resnet_distances_norm, toolbox):
    """
    Calcualte Rank statitics for all images
    """
    func = toolbox.compile(expr=individual)

    def apply_func(row):
        return func(*row)

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.apply(
        apply_func, axis=1, raw=True
    )

    cluster_norm_distances.sort_values(
        by="dlib_distance", inplace=True, ascending=True, ignore_index=True
    )
    by_comb_distances = cluster_norm_distances.sort_values(
        by="combination", ascending=True, ignore_index=True
    )

    imgs = cluster_norm_distances.img1.unique()
    corrs = []
    for img in imgs:
        dlib_img_distances = cluster_norm_distances[
            cluster_norm_distances.img1 == img
        ].reset_index(drop=True)
        comb_img_distances = by_comb_distances[
            by_comb_distances.img1 == img
        ].reset_index(drop=True)

        tmp_corr = dlib_img_distances.img2.corr(
            comb_img_distances.img2, method="kendall"
        )

        if not np.isnan(tmp_corr):
            corrs.append(tmp_corr)

    return min(corrs), max(corrs), np.median(corrs), np.mean(corrs)


def gen_imgs_ranks(
    individual, cluster_norm_distances, resnet_distances_norm, toolbox
) -> pd.DataFrame:
    """
    Calcualte Rank dataframe for all images
    """
    func = toolbox.compile(expr=individual)

    def apply_func(row):
        return func(*row)

    cluster_norm_distances.loc[:, "combination"] = resnet_distances_norm.apply(
        apply_func, axis=1, raw=True
    )

    cluster_norm_distances.sort_values(
        by="dlib_distance", inplace=True, ascending=True, ignore_index=True
    )
    by_comb_distances = cluster_norm_distances.sort_values(
        by="combination", ascending=True, ignore_index=True
    )

    imgs = cluster_norm_distances.img1.unique()
    corrs = []
    for img in imgs:
        dlib_img_distances = cluster_norm_distances[
            cluster_norm_distances.img1 == img
        ].reset_index(drop=True)
        comb_img_distances = by_comb_distances[
            by_comb_distances.img1 == img
        ].reset_index(drop=True)

        tmp_corr = dlib_img_distances.img2.corr(
            comb_img_distances.img2, method="kendall"
        )

        if not np.isnan(tmp_corr):
            corrs.append({"img": img, "rank": tmp_corr})

    return pd.DataFrame(corrs)


# GP Functions
# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def protectSqrt(x):
    return math.sqrt(abs(x))


def log2(x):
    return math.log(abs(x)) if x else 0


def log10(x):
    return math.log10(abs(x)) if x else 0


def pow2(x):
    try:
        return x**2
    except OverflowError:
        return inf


def pow3(x):
    try:
        return x**3
    except OverflowError:
        return inf


def pow4(x):
    try:
        return x**4
    except OverflowError:
        return inf


def _2pow(x):
    try:
        return 2**x
    except OverflowError:
        return inf


def _3pow(x):
    try:
        return 3**x
    except OverflowError:
        return inf


def _4pow(x):
    try:
        return 4**x
    except OverflowError:
        return inf


def gen_primitive_set() -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", IND_SIZE)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(operator.abs, 1)
    pset.addPrimitive(protectSqrt, 1)
    pset.addPrimitive(log2, 1)
    pset.addPrimitive(log10, 1)
    pset.addPrimitive(pow2, 1)  # x²
    pset.addPrimitive(pow3, 1)  # x³
    pset.addPrimitive(pow4, 1)  # x⁴
    pset.addPrimitive(_2pow, 1)  # 2^x
    pset.addPrimitive(_3pow, 1)  # 3^x
    pset.addPrimitive(_4pow, 1)  # 4^x

    try:
        pset.addEphemeralConstant("rand101", lambda: randint(-1, 1))
    except Exception:
        pass  # Ephemral constants are defined globally and shall be declared only once

    return pset


def gp_result_to_dict(individual: gp.PrimitiveTree) -> dict:
    individual_str_deap = str(individual)
    individual_str_readeable = individual_str_deap

    for deap_arg, resnet_col in DEAP_ARGS_FROM_TO.items():
        individual_str_readeable = individual_str_readeable.replace(
            deap_arg, resnet_col
        )

    return {
        "fitness": individual.fitness.values[0],
        "best_deap": individual_str_deap,
        "best_readable": individual_str_readeable,
        "DEAP_ARGS_FROM_TO": DEAP_ARGS_FROM_TO,
    }


def run_experiment():

    with open(RESULTS_FILE, "w") as f:
        f.write(
            "exp_id,cluster,error_function,total_pairs,total_persons,cxpb,mtpb,indpb,pop_size,max_generations,no_best_max_gens,best_generation,best_fitness,min_rank,max_rank,median_rank,mean_rank,exec_time_sec\n"
        )

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

    send_simple_message(
        f"Starting DLIB ResNET GP Experiments with {len(params_comb)} combination of parameters"
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
            exp_id += 1
            cluster_distances = distances[
                (distances.img1_cluster == cluster)
                & (distances.img2_cluster == cluster)
            ]

            cluster_distances = cluster_distances.iloc[:SUB_SET_SIZE]

            total_pairs = len(cluster_distances)
            total_persons = cluster_distances.person1.shape[0]
            print(
                f"""
                    Experiment {exp_id} with {total_pairs} pairs of images of {total_persons} persons.
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

            # Normalize numerical col
            for col in resnet_cols + ["dlib_distance"]:
                cluster_norm_distances[col] = (
                    cluster_norm_distances[col] - cluster_norm_distances[col].min()
                ) / (
                    cluster_norm_distances[col].max()
                    - cluster_norm_distances[col].min()
                )

            resnet_distances_norm = cluster_norm_distances.loc[:, resnet_cols]

            # Prepare DEAP
            pset = gen_primitive_set()

            creator.create(
                "FitnessMin", base.Fitness, weights=(-1.0,)
            )  # Error (minimize)
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
            toolbox.register(
                "individual", tools.initIterate, creator.Individual, toolbox.expr
            )
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("compile", gp.compile, pset=pset)

            creator.create(
                "FitnessMin", base.Fitness, weights=(-1.0,)
            )  # Error (minimize)
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox.register(
                "evaluate",
                current_error_fun,
                cluster_norm_distances=cluster_norm_distances,
                resnet_distances_norm=resnet_distances_norm,
                toolbox=toolbox,
            )
            toolbox.register("select", tools.selTournament, tournsize=3)
            toolbox.register("mate", gp.cxOnePoint)
            toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
            toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

            toolbox.decorate(
                "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
            )
            toolbox.decorate(
                "mutate",
                gp.staticLimit(key=operator.attrgetter("height"), max_value=17),
            )

            seed(RAND_SEED)

            pop = toolbox.population(n=current_pop_size)
            hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(len)
            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
            mstats.register("avg", np.mean)
            mstats.register("std", np.std)
            mstats.register("min", np.min)
            mstats.register("max", np.max)

            start_time = time()

            pop, log = algorithms.eaSimple(
                population=pop,
                toolbox=toolbox,
                cxpb=current_cxpb,
                mutpb=current_mutpb,
                ngen=current_max_generations,
                stats=mstats,
                halloffame=hof,
                verbose=True,
            )

            best = hof[0]

            # Output files for best individuals
            individuals_folder = RESULTS_FOLDER.joinpath(
                f"{str(exp_id).zfill(5)}_individuals"
            )
            individuals_folder.mkdir(exist_ok=True)
            best_individual_file = individuals_folder.joinpath("best_individual.json")
            best_individuals_file = individuals_folder.joinpath("best_individuals.json")
            search_log_file = individuals_folder.joinpath("log.json")
            best_individuals_imgs_ranks_file = individuals_folder.joinpath(
                "imgs_ranks.csv"
            )

            # Save results to files
            json.dump(gp_result_to_dict(best), open(best_individual_file, "w"))
            json.dump(
                [gp_result_to_dict(individual) for individual in hof],
                open(best_individuals_file, "w"),
            )
            json.dump(log, open(search_log_file, "w"))

            # Calculate ranks
            min_rank, max_rank, median_rank, mean_rank = calc_rank(
                best, cluster_norm_distances, resnet_distances_norm, toolbox
            )

            tmp_imgs_ranks = gen_imgs_ranks(
                best, cluster_norm_distances, resnet_distances_norm, toolbox
            )
            tmp_imgs_ranks.to_csv(best_individuals_imgs_ranks_file, index=False)

            with open(RESULTS_FILE, "a") as f:
                tmp_line = f"{exp_id},{cluster},{current_error_fun.__name__},{total_pairs},{total_persons},{current_cxpb},{current_mutpb}"
                tmp_line += f",{current_indpb},{current_pop_size},{current_max_generations},{NO_BEST_MAX_GENERATIONS},{log[-1]['gen']},{best.fitness.values[0]}"
                tmp_line += f",{min_rank},{max_rank},{median_rank},{mean_rank},{int(time()-start_time)}\n"
                f.write(tmp_line)

        if params_experimented % 10 == 0:
            print(
                f"DLIB ResNET GP Experiments:  {params_experimented}/{len(params_comb)} {round(100*params_experimented/len(params_comb),2)}% | Spent {round((time()-params_start_time)//60,2)} min"
            )
            _ = send_simple_message(
                f"DLIB ResNET GP Experiments:  {params_experimented}/{len(params_comb)} {round(100*params_experimented/len(params_comb),2)}% | Spent {round((time()-params_start_time)//60,2)} min"
            )


run_experiment()
