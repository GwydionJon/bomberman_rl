import os
import pickle
import random
import json
from this import d
import numpy as np


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
    self.distance_trace = []
    self.bomb_trace = []
    # to check if the number of coins has changed.
    self.last_coin_number = 0

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
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        with open("model_dict.json", "r") as f:
            self.feature_dict = json.load(f)


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
            print("try")
            decision = np.argmax(
                self.model[feature_index(self, state_to_features(self, game_state)), :]
            )

            self.trace.append(decision)

            self.logger.debug("Model Decision: " + ACTIONS[decision])
            return ACTIONS[decision]

        except:
            print("exept")
            random_action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

            self.trace.append(ACTIONS.index(random_action))
            return random_action


def find_coin(
    self,
    game_state: dict,
):  # Function which finds nearest coin and decides then where to go
    _, score, bool_bomb, (x, y) = game_state["self"]
    current = np.array([x, y])  # Curent position of agent
    coins = game_state["coins"]  # Position of visible coins

    if len(coins) == 0:
        action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        self.distance_trace.append(-1)
        self.stuck = 1
        return ACTIONS.index(action)
    self.stuck = 0
    # if the number of coins has changed we need to choose a new target
    if self.last_coin_number != len(coins):
        self.target = None
    self.last_coin_number = len(coins)

    # calculate nearest target
    min_distance = np.argmin(np.sum(np.abs(coins - current), axis=1))
    # choose target
    if self.target is None:
        self.target = np.array(
            coins[min_distance]
        )  # Setting coordinates of coin/target

    # array of possible movements - free tiles :UP - RIGHT - DOWN - LEFT
    field = game_state["field"].T

    neighbour_tiles = np.array(
        [
            [x, y + 1] if field[x, y + 1] == 0 else [1000, 1000],
            [x + 1, y] if field[x + 1, y] == 0 else [1000, 1000],
            [x, y - 1] if field[x, y - 1] == 0 else [1000, 1000],
            [x - 1, y] if field[x - 1, y] == 0 else [1000, 1000],
        ]
    )

    new_pos = np.argmin(np.sum(np.abs(neighbour_tiles - self.target), axis=1))

    int_distance = int(min_distance / 10)
    if int_distance <= 1:
        self.distance_trace.append(0)
    if int_distance <= 2:
        self.distance_trace.append(1)
    elif int_distance <= 4:
        self.distance_trace.append(2)
    else:
        self.distance_trace.append(3)

    return new_pos  # new acceptable position -> go there


def find_objects(self, object_coordinates, current_pos, field, return_coords=False):
    x, y = current_pos
    if object_coordinates == []:
        if return_coords:
            return 0, 0, None
        return 0, 0
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

    surroundings = [
        field[x, y + 1],
        field[x + 1, y],
        field[x, y - 1],
        field[x - 1, y],
    ]

    if self.trace[-1] == 5:
        self.dropped_bomb = 1
    else:
        self.dropped_bomb = 0

    # is box adjecant:
    self.next_to_box = 0
    if 1 in surroundings:
        self.next_to_box = 1

    if self.target is not None:
        if self.target not in game_state["coins"]:
            self.target = None
    # find coin target
    if self.target is None:
        coin_distance, coin_direction, self.target = find_objects(
            self,
            game_state["coins"],
            (
                x,
                y,
            ),
            field,
            return_coords=True,
        )
    else:
        coin_distance, coin_direction, self.target = find_objects(
            self,
            [self.target],
            (
                x,
                y,
            ),
            field,
            return_coords=True,
        )

    self.distance_trace.append(coin_distance)
    if len(self.distance_trace) > 2:
        # the closer the higher the value
        moved_to_coin = self.distance_trace[-2] < self.distance_trace[-1]
    else:
        moved_to_coin = 0

    # find bomb danger
    bombs_coordinates = [(x, y) for ((x, y), t) in game_state["bombs"]]
    bomb_distance, bomb_direction = find_objects(
        self,
        bombs_coordinates,
        (
            x,
            y,
        ),
        field,
    )

    self.bomb_trace.append(bomb_distance)
    if len(self.bomb_trace) > 2:
        # the closer the higher the value

        moved_away = self.bomb_trace[-2] > self.bomb_trace[-1]
    else:
        moved_away = 0

    features = np.array(
        surroundings
        + [
            moved_to_coin,
            coin_direction,
            moved_away,
            bomb_direction,
            self.bool_bomb,
        ]
    )

    return str(features)


def feature_index(self, features):
    self.feature_dict.setdefault(features, len(self.feature_dict))
    return self.feature_dict[features]
