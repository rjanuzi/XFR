# # Face Recognition (FR) - DLIB ResNET Approximation with Genetic Algorithm

import json
from math import inf
from pathlib import Path
from random import random
from time import time

import numpy as np
import pandas as pd
from deap import base, creator, tools

from util._telegram import send_simple_message

# ### Prepare data and generate extra information


EXPERIMENT_ID = 4
DLIB_DISTANCES_FILE = Path("fr", "distances_dlib.json")
RESNET_DISTANCES_FILE = Path("fr", "distances_resnet.json")
RESNET_FACEPARTS_DISTANCES_FILE = Path("fr", "distances_resnet_faceparts.json")
BEST_INDIVIDUAL_FILE = Path(
    "fr",
    "best_combination_runs",
    f"dlib_resnet_best_comb_{str(EXPERIMENT_ID).zfill(4)}.json",
)
BEST_INDIVIDUALS_FILE = Path(
    "fr",
    "best_combination_runs",
    f"dlib_resnet_best_combs_{str(EXPERIMENT_ID).zfill(4)}.json",
)

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

# ResNET Faceparts Distances
def rows_generator(resnet_faceparts_raw_data):
    for pair, distances in resnet_faceparts_raw_data.items():
        distances.update({"pair": pair})
        yield distances


tmp_raw_data = json.load(open(RESNET_FACEPARTS_DISTANCES_FILE, "r"))
generator = rows_generator(tmp_raw_data)
del tmp_raw_data

resnet_faceparts_distances = pd.DataFrame(generator)

# Join distances into a sigle dataframe
distances = dlib_distances.merge(resnet_faceparts_distances, on="pair", how="outer")

del dlib_distances
del resnet_faceparts_distances

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


# Genetic Algorithm (GA) Search

RESNET_COLS_TO_IGNORE = [
    "resnet_left_ear",
    "resnet_right_ear",
    "resnet_ears",
    "resnet_full_face",
]

# Individuals representation
resnet_cols = list(
    filter(
        lambda c: ("resnet" in c) and (c not in RESNET_COLS_TO_IGNORE),
        distances.columns,
    )
)

IND_SIZE = len(resnet_cols)


# AG Search Params
SUB_SET_SIZE = 5000000  # Number of distances to consider
CXPB = 0.5  # Probability with which two individuals are crossed
MUTPB = 0.3  # Probability for mutating an individual
INDPB = 0.1  # Probability for flipping a bit of an individual
POP_SIZE = 300
MAX_GENERATIONS = 100
NO_BEST_MAX_GENERATIONS = (
    15  # Stop search if have a specific number of generations without improvement
)


cleared_distances = distances.replace(inf, np.nan)
cleared_distances.dropna(inplace=True)
# cleared_distances = cleared_distances[cleared_distances.dlib_distance > 0.01].reset_index(drop=True)
cleared_distances = cleared_distances[
    cleared_distances.img1 != cleared_distances.img2
]  # Remove same image pairs
cleared_distances.sort_values(by="dlib_distance", ascending=True, inplace=True)

sub_df = cleared_distances.iloc[:SUB_SET_SIZE]

# Normalize distances
sub_df = sub_df.loc[
    :, resnet_cols + ["dlib_distance"]
]  # Get numerical columns to nomrlize
for col in sub_df.columns:
    sub_df[col] = (sub_df[col] - sub_df[col].min()) / (
        sub_df[col].max() - sub_df[col].min()
    )

resnet_distances_norm = sub_df.loc[:, resnet_cols]


# Fitness Function

# def eval_individual_error(individual):
#     """
#     Calculate the Mean Squeare Error (MSE) of the individual as a measure of fitness
#     """
#     individual_sum = sum(individual)
#     individual = [i/individual_sum for i in individual]

#     sub_df.loc[:, 'combination'] = resnet_distances_norm.dot(individual)
#     sub_df.loc[:, 'error'] = sub_df.combination - sub_df.dlib_distance
#     sub_df.loc[:, 'sqr_error'] = (sub_df.error.abs()+1) ** 2 # Avoid squared of fractions

#     return (sub_df[sub_df.sqr_error != inf].sqr_error.mean(),) # Shall return a tuple for compatibility with DEAP

# def eval_individual_error(individual):
#     """
#     Calculate the Mean Absolute Error (MAE) of the individual as a measure of fitness
#     """

#     individual_sum = sum(individual)
#     individual = [i/individual_sum for i in individual]

#     sub_df.loc[:, 'combination'] = resnet_distances_norm.dot(individual)
#     sub_df.loc[:, 'error'] = sub_df.combination - sub_df.dlib_distance

#     return (sub_df[sub_df.error != inf].error.abs().mean(),) # Shall return a tuple for compatibility with DEAP

# def eval_individual_error(individual):
#     """
#     Calculate the Absolute Error Sum of the individual as a measure of fitness
#     """

#     individual_sum = sum(individual)
#     individual = [i/individual_sum for i in individual]

#     sub_df.loc[:, 'combination'] = resnet_distances_norm.dot(individual)
#     sub_df.loc[:, 'error'] = sub_df.combination - sub_df.dlib_distance

#     return (sub_df[sub_df.error != inf].error.abs().sum(),) # Shall return a tuple for compatibility with DEAP


def eval_individual_error(individual):
    """
    Calculate the Step differente of the individual as a measure of fitness
    """

    individual_sum = sum(individual)
    if individual_sum == 0:
        return (inf,)
    individual = [i / individual_sum for i in individual]

    sub_df.loc[:, "combination"] = resnet_distances_norm.dot(individual)

    # Pandas Like Error
    sub_df.loc[:, "dlib_same_person"] = sub_df.dlib_distance.apply(
        lambda c: 1 if c < 0.37 else 0
    )
    sub_df.loc[:, "comb_same_person"] = sub_df.combination.apply(
        lambda c: 1 if c < 0.37 else 0
    )
    sub_df.loc[:, "error"] = sub_df.comb_same_person - sub_df.dlib_same_person

    # Shall return a tuple for compatibility with DEAP
    return (sub_df[sub_df.error != inf].error.abs().sum(),)


# creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Corr (maximize)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Error (minimize)
# creator.create("Individual", list, fitness=creator.FitnessMax)
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_individual_error)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=3)


start_time = time()
pop = toolbox.population(n=POP_SIZE)

fitness_time = time()
# Evaluate the entire population
for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
    ind.fitness.values = fit

# Extracting all the fitnesses of
fits = [ind.fitness.values[0] for ind in pop]

# Variable keeping track of the number of generations
g = 0

low_std_times = 0
last_max_fit = -1e9
last_min_fit = +1e9
best_generation = 0
bests = []

send_simple_message("Starting GA")

# Begin the evolution
while g < MAX_GENERATIONS:
    # A new generation
    g = g + 1

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random() < CXPB:
            toolbox.mate(child1, child2)
            # Invalidate fitnesses for the new individuals
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random() < MUTPB:
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
    std = abs(sum2 / length - mean ** 2) ** 0.5

    if std < 3e-3:
        low_std_times += 1

    # If we face 5 generations with low standard deviation, let's resset population
    if low_std_times > 5:
        low_std_times = 0

        # Reset Pop
        print("Resetting pop")
        send_simple_message("Resetting pop")
        pop = toolbox.population(n=POP_SIZE)

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
        send_simple_message(f"New best found (Gen: {best_generation}): {last_min_fit}")
        json.dump(dict(zip(resnet_cols, best)), open(BEST_INDIVIDUAL_FILE, "w"))

        bests.append(
            {
                "generation": best_generation,
                "fitness": last_min_fit,
                "best_data": dict(zip(resnet_cols, best)),
            }
        )
        json.dump(bests, open(BEST_INDIVIDUALS_FILE, "w"))

    if g - best_generation > NO_BEST_MAX_GENERATIONS:
        print("No best found for too long, ending search")
        send_simple_message("No best found for too long, ending search")
        break

    print(
        f"{g} Gens | Best fitness: {last_min_fit} | Std: {std} | {(time() - start_time)//60} min | {g}/{MAX_GENERATIONS} | ({round(g/MAX_GENERATIONS*100, 2)} %)"
    )
    _ = send_simple_message(
        f"{g} Gens | Best fitness: {last_min_fit} | Std: {std} | {(time() - start_time)//60} min | {g}/{MAX_GENERATIONS} | ({round(g/MAX_GENERATIONS*100, 2)} %)"
    )


print(
    f"Done: {g} generations. Best fitness: {last_min_fit} at generation {best_generation} in {(time() - start_time)//60} minutes"
)
_ = send_simple_message(
    f"Done: {g} generations. Best fitness: {last_min_fit} at generation {best_generation} in {(time() - start_time)//60} minutes"
)
