import os
import pickle
import random
import json
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import settings as s

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
STATE_FEATURES = 27
model_path = "tensorflow_model"


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
    self.target = None
    self.running_circles = 0

    # stores the distance to the target
    self.trace = []
    self.distance_trace = []
    self.bomb_trace = []
    # to check if the number of coins has changed.
    self.last_coin_number = 0

    if self.train and not os.path.isfile("model_path"):
        print("First round")
        self.logger.info("Setting up model from scratch.")
        self.first_training_round = True

    elif self.train and os.path.isfile("model_path"):
        print("second round")
        self.logger.info("Loading model from saved state.")
        self.first_training_round = False
        self.model = load_model(model_path)

    else:
        self.first_training_round = False
        self.logger.info("Loading model from saved state.")
        self.model = load_model(model_path)

    # target stores the current target coin
    self.target = None
    # trace is a list of all decisions
    self.trace = []
    # stores the distance to the target
    self.distance_trace = []

    # to check if the number of coins has changed.
    self.last_coin_number = 0
    if self.train:  # or not os.path.isfile(model_path):
        print("no model yet")

    else:
        self.logger.info("Loading model from saved state.")
        # with open("my-saved-model.pt", "rb") as file:
        print("load")


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

        feature = state_to_features(self, game_state).reshape(-1, STATE_FEATURES)

        decision = self.model.predict(feature)

        self.logger.debug(
            "Model Decision: " + str(np.argmax(decision)) + "from : " + str(decision)
        )

        return ACTIONS[np.argmax(decision)]


def find_objects(self, object_coordinates, current_pos, field, return_coords=False):
    x, y = current_pos
    if object_coordinates == []:
        if return_coords:
            return 0, -1, None
        return 0, -1
    min_distance_ind = np.argmin(
        np.sum(np.abs(np.asarray(object_coordinates) - np.asarray(current_pos)), axis=1)
    )
    object_coord = object_coordinates[min_distance_ind]

    distance = np.sum(
        np.abs(np.asarray(object_coord) - np.asarray(current_pos)), axis=0
    )
    neighbour_tiles = np.array(
        [
            [x, y + 1] if field[x, y + 1] == 0 else [1000, 1000],
            [x + 1, y] if field[x + 1, y] == 0 else [1000, 1000],
            [x, y - 1] if field[x, y - 1] == 0 else [1000, 1000],
            [x - 1, y] if field[x - 1, y] == 0 else [1000, 1000],
        ]
    )

    direction = np.argmin(np.sum(np.abs(neighbour_tiles - object_coord), axis=1))

    if distance <= 1:
        distance_value = 4
    elif distance <= 2:
        distance_value = 3
    elif distance <= 4:
        distance_value = 2
    else:
        distance_value = 1

    if return_coords:
        return distance_value, direction, object_coord
    return distance_value, direction


def add_bomb_path_to_field(bomb_coords, field):
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

    if bomb_coords == []:
        return field
    for bomb_coord in bomb_coords:
        blast_coord = _get_blast_coords(bomb_coord, field)
        for blast in blast_coord:
            field[blast] = 2
    return field


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

    _, score, self.bool_bomb, (x, y) = game_state["self"]
    field = game_state["field"].T
    # check in which directions walls are: UP-RIGHT-DOWN-LEFT

    # surroundings = [
    #     field[x, y + 1],
    #     field[x + 1, y],
    #     field[x, y - 1],
    #     field[x - 1, y],
    # ]
    bombs_coordinates = [(x, y) for ((x, y), t) in game_state["bombs"]]

    field = add_bomb_path_to_field(bombs_coordinates, field)

    # search for bombs, crates and space in surroundings
    self.surroundings = []

    for x_i in range(-2, 3):
        for y_i in range(-2, 3):
            if (x + x_i) < field.shape[0] and (y + y_i) < field.shape[1]:
                self.surroundings.append(field[x + x_i, y + y_i])
            else:
                self.surroundings.append(-1)

    # is box adjecant:
    self.next_to_box = 0
    if 1 in self.surroundings:
        self.next_to_box = 1

    # find coin target
    coin_distance, self.coin_direction, self.target = find_objects(
        self,
        game_state["coins"],
        (
            x,
            y,
        ),
        field,
        return_coords=True,
    )

    bomb_distance, self.bomb_direction = find_objects(
        self,
        bombs_coordinates,
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
            self.bomb_direction,
        ]
    )
    return features
