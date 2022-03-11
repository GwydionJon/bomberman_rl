import os
import pickle
import random
import json

# from this import d
import numpy as np
import settings as s
from collections import namedtuple, deque


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
STATE_FEATURES = 2


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # target stores the current target coin
    self.target = None

    # stores the distance to the target
    self.trace = []
    # to check if the number of coins has changed.

    self.score = deque(maxlen=100)
    with open("rewards.json", "r") as f:
        self.reward_dict = json.load(f)

    if self.train and not os.path.isfile("my-saved-model.pt"):
        print("First round")
        self.logger.info("Setting up model from scratch.")
        self.first_training_round = True
        self.model = np.zeros((STATE_FEATURES, len(ACTIONS)))
        self.feature_dict = {}
    elif self.train and os.path.isfile("my-saved-model.pt"):
        print("second round")
        self.logger.info("Loading model from saved state.")
        self.first_training_round = False
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        with open("model_dict.json", "r") as f:
            self.feature_dict = json.load(f)
    else:
        self.first_training_round = False
        self.logger.info("Loading model from saved state.\n")

        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        with open("model_dict.json", "r") as f:
            self.feature_dict = json.load(f)
        [self.logger.info(str(row)) for row in self.model]


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # best_step = find_coin(self, game_state)

    random_prob = 0.25
    if self.first_training_round is True and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.

        random_action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        self.trace.append(ACTIONS.index(random_action))
        return random_action
    else:
        self.logger.debug("Querying model for action.")

        try:

            decision = np.argsort(
                self.model[feature_index(self, state_to_features(self, game_state)), :],
                axis=0,
            )

            # decision = np.random.choice(
            #     [decision[-1], decision[-2]],
            #     p=[
            #         decision[-1] / (decision[-1] + decision[-2]),
            #         decision[-2] / (decision[-1] + decision[-2]),
            #     ],
            # )
            decision = decision[-1]

            self.trace.append(decision)

            self.logger.debug(
                "Model Decision: "
                + str(decision)
                + " chosen from:"
                + str(
                    self.model[
                        feature_index(self, state_to_features(self, game_state)), :
                    ]
                )
            )
            return ACTIONS[decision]

        except:
            print("exept")
            random_action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

            self.trace.append(ACTIONS.index(random_action))
            return random_action


def find_objects(self, object_coordinates, current_pos, field, return_coords=False):
    x, y = current_pos
    if object_coordinates is not False:
        if return_coords:
            return 0, -1, None
        return 0, -1
    else:
        min_distance_ind = np.argmin(
            np.sum(
                np.abs(np.asarray(object_coordinates) - np.asarray(current_pos)), axis=1
            )
        )

        object_coord = object_coordinates[min_distance_ind]

        distance = np.linalg.norm(np.asarray(object_coord) - np.asarray(current_pos))
        neighbour_tiles = np.array(
            [
                [x, y + 1] if field[x, y + 1] == 0 else [1000, 1000],
                [x + 1, y] if field[x + 1, y] == 0 else [1000, 1000],
                [x, y - 1] if field[x, y - 1] == 0 else [1000, 1000],
                [x - 1, y] if field[x - 1, y] == 0 else [1000, 1000],
            ]
        )

        direction = np.argmin(np.sum(np.abs(neighbour_tiles - object_coord), axis=1))

        if return_coords:
            return distance, direction, object_coord
        return distance, direction


def add_bomb_path_to_field(bombs, explosions, field):
    def _get_blast_coords(bomb, field):
        x, y = bomb[0], bomb[1]
        blast_coords = [(x, y)]

        for i in range(1, s.BOMB_POWER + 1):
            if field[x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, s.BOMB_POWER + 1):
            if field[x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, s.BOMB_POWER + 1):
            if field[x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, s.BOMB_POWER + 1):
            if field[x, y - i] == -1:
                break
            blast_coords.append((x, y - i))

        return blast_coords

    # print(explosions)
    bomb_coords = [(x, y) for ((x, y), t) in bombs]
    if bomb_coords == []:
        return field
    for bomb_coord in bomb_coords:
        blast_coord = _get_blast_coords(bomb_coord, field)
        for blast in blast_coord:
            field[blast] = 2
        field[np.where(explosions == 1)] = 2

    # also add current explosion map

    return field


def find_safe_spot(self, field, position):
    print(field)
    x, y = position
    close_field = np.zeros((7, 7))
    for i in range(-3, 4):
        for j in range(-3, 4):
            if ((x + i) < 0 or (x + i) >= len(field)) or (
                (y + j) < 0 or (y + j) >= len(field)
            ):
                close_field[i + 3, j + 3] = -1
            else:
                close_field[i + 3, j + 3] = field[x + i, y + j]
    print(close_field)
    print(np.where(close_field == 0))


def find_crates(self, field, position):

    field_array = np.asarray(field)
    crate_coords = np.asarray(np.where(field_array == 1)).T
    return crate_coords


def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends

    if game_state is None:
        return None
    # get all coordinates transpoed
    _, score, self.bool_bomb, position = game_state["self"]

    field = game_state["field"].T
    (y, x) = position
    bombs = [((y, x), t) for ((x, y), t) in game_state["bombs"]]
    bomb_coordinates = [(y, x) for ((x, y), t) in game_state["bombs"]]
    explosion_map = game_state["explosion_map"].T
    coins = np.asarray(game_state["coins"]).T

    # check in which directions walls are: UP-RIGHT-DOWN-LEFT

    field = add_bomb_path_to_field(bombs, explosion_map, field)

    # search for bombs, crates and space in surroundings
    self.surroundings = []

    for i in range(-2, 3):
        if (x + i) < field.shape[0]:
            self.surroundings.append(field[x + i, y])
        else:
            self.surroundings.append(-1)

        if (y + i) < field.shape[1]:
            self.surroundings.append(field[x, y + i])
        else:
            self.surroundings.append(-1)

    save_direction = find_safe_spot(self, field, (x, y))

    # find coin target
    coin_distance, self.coin_direction, self.target = find_objects(
        self,
        coins,
        (
            x,
            y,
        ),
        field,
        return_coords=True,
    )

    # bombs_coordinates = [(x, y) for ((x, y), t) in game_state["bombs"]]
    bomb_distance, self.bomb_direction = find_objects(
        self,
        bomb_coordinates,
        (
            x,
            y,
        ),
        field,
    )
    crate_coords = find_crates(self, field, (x, y))

    crate_distance, crate_direction = find_objects(
        self,
        crate_coords,
        (
            x,
            y,
        ),
        field,
    )

    features = np.array(
        self.surroundings
        + [
            self.coin_direction,
            crate_direction,
            self.bool_bomb,
        ]
    )
    return str(features)


def feature_index(self, features):
    self.feature_dict.setdefault(features, len(self.feature_dict))
    return self.feature_dict[features]
