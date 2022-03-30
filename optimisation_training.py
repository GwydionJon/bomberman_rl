import os
import json
import numpy as np
from events import CRATE_DESTROYED
import pygad
import pickle

# @misc{gad2021pygad,

#       title={PyGAD: An Intuitive Genetic Algorithm Python Library},

#       author={Ahmed Fawzy Gad},

#       year={2021},

#       eprint={2106.06158},

#       archivePrefix={arXiv},

#       primaryClass={cs.NE}

# }

# N x M (number of feature combinations x number of actions)
N = 960
M = 6


model_name = "cc_agent_genetic_tuning"
model_folder = os.path.join("agent_code", model_name)


num_genes = N * M  # number of parameters

num_generations = 2
num_parents_mating = 2

sol_per_pop = 3  # solutions per generation

init_range_low = -20
init_range_high = 20

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10


def make_example_dict(values):
    model = np.asarray(values).reshape((N, M))

    return model


def find_model(values, value_idx):
    return run_model(values, value_idx)


def run_model(values, value_idx, keep_model=False, play=False):

    model = make_example_dict(values)
    with open(
        os.path.join(model_folder, "model_creation/initial_model.pt"), "wb"
    ) as file:
        pickle.dump(model, file)

    if play == False:
        os.system(
            "python main.py play --agent "
            + model_name
            + " --train 1 --no-gui --n-rounds 100"
        )
    else:
        os.system("python main.py play --agent " + model_name)

    with open(os.path.join(model_folder, "model_testing/stat.json"), "r") as f:
        stats_dict = json.load(f)

    if keep_model == False:
        # os.remove(os.path.join(model_folder, "model_testing/stat.json"))
        os.remove(os.path.join(model_folder, "model_creation/initial_model.pt"))
        os.remove(os.path.join(model_folder, "model_running/my-saved-model.pt"))
    return stats_dict["0"]


def init_pygad():

    fitness_function = find_model

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
    )

    return ga_instance


def main():

    ga_instance = init_pygad()
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(
        "Parameters of the best solution : {solution}".format(
            solution=make_example_dict(solution)
        )
    )
    print(
        "Fitness value of the best solution = {solution_fitness}".format(
            solution_fitness=solution_fitness
        )
    )
    ga_instance.plot_fitness()
    # train the best model
    run_model(solution, 0, keep_model=True)
    run_model(solution, 0, keep_model=True, play=True)


if __name__ == "__main__":
    main()
