import numpy as np
import random as random
import json
import pickle
import os


def get_feature_dict():
    print("intilize")

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

    return feature_dict


def main():
    feature_dict = get_feature_dict()
    path = os.path.join(os.path.dirname(__file__), "feature_dict.json")
    with open(path, "w") as fp:
        json.dump(feature_dict, fp)


if __name__ == "__main__":
    main()
