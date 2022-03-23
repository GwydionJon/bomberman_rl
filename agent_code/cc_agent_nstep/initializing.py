import numpy as np
import random as random
import json
import pickle

feature_dict = {}
counter = 0
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                for m in range(-1, 4):
                    for n in range(2):
                        for o in range(-2, 4):
                            features = np.array([i, j, k, l, m, n, o])

                            feature_dict.setdefault(str(features), counter)
                            counter += 1

with open("feature_dict.json", "w") as fp:
    json.dump(feature_dict, fp)


def create_table_v1(feature_dict):
    N = len(feature_dict)
    M = 6  # Number of actions
    table = np.random.randint(1, 20, (N, M))

    for m in range(-1, 4):
        for n in range(2):
            for o in range(-2, 4):
                corner1 = feature_dict[str(np.array([1, 1, 0, 0, m, 1, o]))]
                corner2 = feature_dict[str(np.array([0, 1, 1, 0, m, 1, o]))]
                corner3 = feature_dict[str(np.array([0, 0, 1, 1, m, 1, o]))]
                corner4 = feature_dict[str(np.array([1, 0, 0, 1, m, 1, o]))]

                table[corner1] = [-100, -100, 100, 100, 0, -200]
                table[corner2] = [100, -100, -100, 100, 0, -200]
                table[corner3] = [100, 100, -100, -100, 0, -200]
                table[corner4] = [-100, 100, 100, -100, 0, -200]

                updown = feature_dict[str(np.array([1, 0, 1, 0, m, n, o]))]
                leftright = feature_dict[str(np.array([0, 1, 0, 1, m, n, o]))]

                table[updown, 0] = -100
                table[updown, 2] = -100
                table[leftright, 1] = -100
                table[leftright, 3] = -100

                three_stop1 = feature_dict[str(np.array([1, 1, 1, 0, m, n, o]))]
                three_stop2 = feature_dict[str(np.array([0, 1, 1, 1, m, n, o]))]
                three_stop3 = feature_dict[str(np.array([1, 0, 1, 1, m, n, o]))]
                three_stop4 = feature_dict[str(np.array([1, 1, 0, 1, m, n, o]))]

                # three waslls and already placed bomb
                three_stop1wb = feature_dict[str(np.array([1, 1, 1, 0, m, 0, o]))]
                three_stop2wb = feature_dict[str(np.array([0, 1, 1, 1, m, 0, o]))]
                three_stop3wb = feature_dict[str(np.array([1, 0, 1, 1, m, 0, o]))]
                three_stop4wb = feature_dict[str(np.array([1, 1, 0, 1, m, 0, o]))]

                table[three_stop1, 0] = -100
                table[three_stop1, 1] = -100
                table[three_stop1, 2] = -100
                table[three_stop2, 1] = -100
                table[three_stop2, 2] = -100
                table[three_stop2, 3] = -100
                table[three_stop3, 0] = -100
                table[three_stop3, 2] = -100
                table[three_stop3, 3] = -100
                table[three_stop4, 0] = -100
                table[three_stop4, 1] = -100
                table[three_stop4, 3] = -100

                # Always a good bomb placement since max two walls and one must be crate
                table[three_stop1, -1] = 250
                table[three_stop2, -1] = 250
                table[three_stop3, -1] = 250
                table[three_stop4, -1] = 250

                # As bomb already placed should not place another
                table[three_stop1wb, -1] = -250
                table[three_stop2wb, -1] = -250
                table[three_stop3wb, -1] = -250
                table[three_stop4wb, -1] = -250

                three_open1 = feature_dict[str(np.array([1, 0, 0, 0, m, n, o]))]
                three_open2 = feature_dict[str(np.array([0, 1, 0, 0, m, n, o]))]
                three_open3 = feature_dict[str(np.array([0, 0, 1, 0, m, n, o]))]
                three_open4 = feature_dict[str(np.array([0, 0, 0, 1, m, n, o]))]

                table[three_open1, 0] = -100
                table[three_open2, 1] = -100
                table[three_open3, 2] = -100
                table[three_open4, 3] = -100

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(-1, 4):
                        for o in range(-2, 4):
                            # walking in the safe direction
                            safe1 = feature_dict[str(np.array([i, j, k, l, m, 0, 0]))]
                            safe2 = feature_dict[str(np.array([i, j, k, l, m, 0, 1]))]
                            safe3 = feature_dict[str(np.array([i, j, k, l, m, 0, 2]))]
                            safe4 = feature_dict[str(np.array([i, j, k, l, m, 0, 3]))]

                            table[safe1, 0] = 100
                            table[safe2, 1] = 100
                            table[safe3, 2] = 100
                            table[safe4, 3] = 100

                            # coin direction encouraged
                            coin1 = feature_dict[str(np.array([i, j, k, l, 0, 1, o]))]
                            coin2 = feature_dict[str(np.array([i, j, k, l, 1, 1, o]))]
                            coin3 = feature_dict[str(np.array([i, j, k, l, 2, 1, o]))]
                            coin4 = feature_dict[str(np.array([i, j, k, l, 3, 1, o]))]

                            table[coin1, 0] = 130
                            table[coin2, 1] = 130
                            table[coin3, 2] = 130
                            table[coin4, 3] = 130

    return table


def create_table(feature_dict):
    N = len(feature_dict)
    M = 6  # Number of actions
    table = np.random.randint(1, 20, (N, M))
    return table


table = create_table_v1(feature_dict)

np.savetxt("startingmodel.csv", table, fmt="%s", delimiter=",")

with open("starting_model.pt", "wb") as file:
    pickle.dump(table, file)
