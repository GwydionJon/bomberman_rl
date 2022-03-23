import os
import json
import numpy as np
from events import CRATE_DESTROYED
import pygad


# @misc{gad2021pygad,

#       title={PyGAD: An Intuitive Genetic Algorithm Python Library},

#       author={Ahmed Fawzy Gad},

#       year={2021},

#       eprint={2106.06158},

#       archivePrefix={arXiv},

#       primaryClass={cs.NE}

# }
model_name = "cc_agent_nstep_tuning"
model_folder = os.path.join("agent_code", model_name)


num_genes = 15  # number of parameters

num_generations = 10
num_parents_mating = 5

sol_per_pop = 25  # solutions per generation

init_range_low = 0
init_range_high = 40

parent_selection_type = "sss"
keep_parents = 3

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10


def make_example_dict(values):
    reward_dict = {}

    pos_keys = [
        "COIN_COLLECTED",
        "KILLED_OPPONENT",
        "COIN_FOUND",
        "CRATE_DESTROYED",
        "CHASING_COIN",
        "CHASING_CRATE",
        "GOOD_BOMB_PLACEMENT",
        "MOVED_FROM_DANGER",
        "MOVED_FROM_BOMB",
        "CHASING_COIN_INSTEAD_OF_CRATE",
    ]

    for key, value in zip(pos_keys, values[: len(pos_keys)]):
        reward_dict[key] = float(value)

    neg_keys = [
        "WAITED",
        "INVALID_ACTION",
        "KILLED_SELF",
        "MOVED_TOWARDS_BOMB",
        "BAD_BOMB_PLACEMENT",
        "MOVED_IN_DANGER",
        "CHASING_CRATE_INSTEAD_OF_COIN",
        "SAME_ACTIONS",
        "WAIT_AT_BOMB_POSITION",
    ]

    for key, value in zip(
        neg_keys, values[len(pos_keys) : len(pos_keys) + len(neg_keys)]
    ):
        if key == "KILLED_SELF":
            reward_dict[key] = -float(value) * 1.5
        elif key == "INVALID_ACTION":
            reward_dict[key] = -float(value) * 1.3

        else:
            reward_dict[key] = -float(value)

    movement_keys = [
        "MOVED_DOWN",
        "MOVED_LEFT",
        "MOVED_RIGHT",
        "MOVED_UP",
    ]

    reward_dict["N"] = int((values[-4] / init_range_high) * 10) + 1

    reward_dict["epsilon"] = float(values[-4]) / init_range_high
    reward_dict["alpha"] = float(values[-3]) / init_range_high
    reward_dict["gamma"] = float(values[-2]) / init_range_high

    for key in movement_keys:
        reward_dict[key] = -float(values[-1])

    return reward_dict


def find_model(values, value_idx):
    return run_model(values, value_idx)


def run_model(values, value_idx, keep_model=False):

    reward_dict = make_example_dict(values)
    with open(os.path.join(model_folder, "reward_dict.json"), "w") as fp:
        json.dump(reward_dict, fp)

    os.system(
        "python main.py play --agent "
        + model_name
        + " --train 1 --no-gui --n-rounds 1000"
    )

    with open(os.path.join(model_folder, "stat.json"), "r") as f:
        stats_dict = json.load(f)

    if keep_model == False:
        os.remove(os.path.join(model_folder, "my-saved-model.pt"))

    return stats_dict[str(list(reward_dict.values()))]


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

    # train the best model
    run_model(solution, 0, keep_model=True)


if __name__ == "__main__":
    main()
