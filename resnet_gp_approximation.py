# Face Recognition (FR) - DLIB ResNET Approximation with Genetic Programming

import json
import math
import operator
from pathlib import Path
from random import randint, seed
from time import time

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools

from util._telegram import send_simple_message

### Prepare data and generate extra information


EXPERIMENT_ID = 4


DLIB_DISTANCES_FILE = Path("fr", "distances_dlib.json")
RESNET_DISTANCES_FILE = Path("fr", "distances_resnet.json")
RESNET_FACEPARTS_DISTANCES_FILE = Path("fr", "distances_resnet_faceparts.json")

BEST_INDIVIDUAL_FILE = Path(
    "fr",
    "best_combination_runs",
    f"gp_dlib_resnet_best_comb_{str(EXPERIMENT_ID).zfill(4)}.json",
)
BEST_INDIVIDUALS_FILE = Path(
    "fr",
    "best_combination_runs",
    f"gp_dlib_resnet_best_combs_{str(EXPERIMENT_ID).zfill(4)}.json",
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

# ### Genetic Programming (GP) Search


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


SUB_SET_SIZE = 1000000  # Number of distances to consider
CXPB = 0.5  # Probability with which two individuals are crossed
MUTPB = 0.25  # Probability for mutating an individual
POP_SIZE = 200
HALL_OF_FAME_SIZE = 10
MAX_GENERATIONS = 30


cleared_distances = distances.replace(math.inf, np.nan)
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
    return x ** 2


def pow3(x):
    return x ** 3


def pow4(x):
    return x ** 4


def _2pow(x):
    return 2 ** x


def _3pow(x):
    return 3 ** x


def _4pow(x):
    return 4 ** x


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


pset.addEphemeralConstant("rand101", lambda: randint(-1, 1))


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Error (minimize)
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval_individual_error_gp(individual):
    """
    Calculate the Mean Absolute Error (MAE) of the individual as a measure of fitness
    """
    func = toolbox.compile(expr=individual)

    def apply_func(row):
        return func(*row)

    sub_df.loc[:, "combination"] = resnet_distances_norm.apply(
        apply_func, axis=1, raw=True
    )
    sub_df.loc[:, "error"] = sub_df.combination - sub_df.dlib_distance

    return (
        sub_df[sub_df.error != math.inf].error.abs().mean(),
    )  # Shall return a tuple for compatibility with DEAP


toolbox.register("evaluate", eval_individual_error_gp)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)
toolbox.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)

seed(318)

pop = toolbox.population(n=POP_SIZE)
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
    pop,
    toolbox,
    CXPB,
    MUTPB,
    MAX_GENERATIONS,
    stats=mstats,
    halloffame=hof,
    verbose=True,
)

end_time = time()
print(f"GP finished in {int((end_time - start_time)/60)} minutes")
_ = send_simple_message(f"GP finished in {int((end_time - start_time)/60)} minutes")


best = hof[0]
best_tree = gp.PrimitiveTree(best)
print(str(best_tree))


pop_fitness = pd.DataFrame(
    dict(pop_fitness=np.array([i.fitness.values[0] for i in pop]))
)


hof_fitness = pd.DataFrame(
    dict(hof_fitness=np.array([i.fitness.values[0] for i in hof]))
)
