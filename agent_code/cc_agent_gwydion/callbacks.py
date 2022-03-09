import os
import pickle
import random
import json
import numpy as np


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]  # , 'BOMB']
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
    # trace is a list of all decisions
    self.trace = []
    # stores the distance to the target
    self.distance_trace = []

    # to check if the number of coins has changed.
    self.last_coin_number = 0

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        print("create new model in callbacks")
        self.model = np.zeros((STATE_FEATURES, len(ACTIONS)))
        # feature dict stores different feature combinations and maps them to our model.
        self.feature_dict = {}

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

        print(self.model)
        print(self.model.shape)

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

    random_prob = 0.2
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.2])
    else:
        self.logger.debug("Querying model for action.")

        try:
            self.trace.append(
                np.argmax(
                    self.model[
                        feature_index(self, state_to_features(self, game_state)), :
                    ]
                )
            )
            return ACTIONS[self.trace[-1]]
        except:
            return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.2])


def find_coin(
    self,
    game_state: dict,
):  # Function which finds nearest coin and decides then where to go
    _, score, bool_bomb, (x, y) = game_state["self"]
    current = np.array([x, y])  # Curent position of agent
    coins = game_state["coins"]  # Position of visible coins
    if len(coins) == 0:
        action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        self.distance_trace.append(-1)
        return ACTIONS.index(action)

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
    self.distance_value = 1
    if int_distance <= 1:
        self.distance_trace.append(0)
    if int_distance <= 2:
        self.distance_trace.append(1)
    elif int_distance <= 4:
        self.distance_trace.append(2)
    else:
        self.distance_trace.append(3)

    return new_pos  # new acceptable position -> go there


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

    _, score, bool_bomb, (x, y) = game_state["self"]
    coin_target = find_coin(self, game_state)

    field = game_state["field"].T
    # check in which directions walls are: UP-RIGHT-DOWN-LEFT
    find_walls = [
        1 if field[x, y + 1] == -1 else 0,
        1 if field[x + 1, y] == -1 else 0,
        1 if field[x, y - 1] == -1 else 0,
        1 if field[x - 1, y] == -1 else 0,
    ]

    # add the last step as feature only when previous steps exist.
    if len(self.trace) > 2:
        previous_step = self.trace[-1]
    else:
        previous_step = 0
    features = np.array(find_walls + [coin_target, self.distance_value, previous_step])

    return str(features)


def feature_index(self, features):
    self.feature_dict.setdefault(features, len(self.feature_dict))
    return self.feature_dict[features]
