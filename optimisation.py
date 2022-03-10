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
model_name = "cc_agent_gwydion_tuning"

num_genes = 18  # number of parameters

num_generations = 20
num_parents_mating = 5

sol_per_pop = 40  # solutions per generation

init_range_low = 0
init_range_high = 70

parent_selection_type = "sss"
keep_parents = 2

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
    ]
    for key, value in zip(
        neg_keys, values[len(pos_keys) : len(pos_keys) + len(neg_keys)]
    ):
        reward_dict[key] = -float(value)

    movement_keys = [
        "MOVED_DOWN",
        "MOVED_LEFT",
        "MOVED_RIGHT",
        "MOVED_UP",
    ]

    reward_dict["epsilon"] = float(values[-4]) / init_range_high
    reward_dict["alpha"] = float(values[-3]) / init_range_high
    reward_dict["gamma"] = float(values[-2]) / init_range_high

    for key in movement_keys:
        reward_dict[key] = -float(values[-1])

    return reward_dict


def run_model(values, value_idx):

    reward_dict = make_example_dict(values)
    with open("agent_code/cc_agent_gwydion_Tuning/rewards.json", "w") as fp:
        json.dump(reward_dict, fp)

    os.system(
        "python main.py play --agent cc_agent_gwydion_Tuning --train 1 --no-gui --n-rounds 1000"
    )

    with open("agent_code/cc_agent_gwydion_Tuning/stat.json", "r") as f:
        stats_dict = json.load(f)

    # os.remove("agent_code/cc_agent_gwydion_Tuning/model_dict.json")
    # os.remove("agent_code/cc_agent_gwydion_Tuning/my-saved-model.pt")

    return stats_dict[str(list(reward_dict.values()))]


def init_pygad():

    fitness_function = run_model

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

    # ga_instance = init_pygad()
    # ga_instance.run()

    # #test_run = run_model(np.arange(15))

    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # print("Parameters of the best solution : {solution}".format(solution=make_example_dict(solution)))
    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    solution = [
        12.868106814929837,
        59.13289993245976,
        59.55190778919831,
        6.5840939230389095,
        54.95605131095348,
        6.091471200364805,
        62.236969513572035,
        5.161016394931826,
        1.2407603745344304,
        25.55570619700454,
        41.89826735891162,
        32.50923261654381,
        3.3619782069834736,
        4.4775502503016975,
        13.194939666513676,
        13.194939666513676,
        13.194939666513676,
        13.194939666513676,
        13.194939666513676,
    ]

    run_model(solution, 0)


if __name__ == "__main__":
    main()
