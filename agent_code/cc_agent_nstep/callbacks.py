import os
import pickle
import random
import json

# from this import d
import numpy as np
import settings as s
from collections import namedtuple, deque
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.grid import Grid



ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
#STATE_FEATURES = 2


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
    #with open("rewards.json", "r") as f:
    #    self.reward_dict = json.load(f)
    with open("feature_dict.json", "r") as f:
        self.feature_dict = json.load(f)
        
    if self.train and not os.path.isfile("my-saved-model.pt"):
        print("First round")
        self.logger.info("Setting up model from scratch.")
        self.first_training_round = True
        #self.feature_dict = {}
    elif self.train and os.path.isfile("my-saved-model.pt"):
        print("Second round")
        self.logger.info("Loading model from saved state.")
        self.first_training_round = False
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        #with open("model_dict.json", "r") as f:
        #    self.feature_dict = json.load(f)
    else:
        self.first_training_round = False
        self.logger.info("Loading model from saved state.\n")

        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        
        #[self.logger.info(str(row)) for row in self.model]


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

    features = state_to_features(self, game_state)
    index = feature_index(self, features)

    # random_prob = 0.25
    # if (
    #     self.first_training_round is True and random.random() < random_prob
    # ) or index == -1:
    #     self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.

    #     random_action = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    #     self.trace.append(ACTIONS.index(random_action))
    #     return random_action
    if self.train:
        transitions = np.array(self.transitions)
        #decision = np.argmax(self.model[index, :])
    if self.train and len(self.transitions) >=3 and transitions[-1,1]==transitions[-3,1]:
        self.logger.debug("Trying to break repetetive actions.")
        decision = np.random.choice([0,1,2,3,4,5], p=[0.25, 0.25, 0.25, 0.25, 0, 0])
    else:
        self.logger.debug("Querying model for action.")
    
        decision = np.argmax(self.model[index, :])

        #decision = decisions[-1]
        #self.trace.append(decision)
        #(ACTIONS[decision])
    self.logger.debug(
            "Model Decision: "
            + str(decision)
            + " chosen from:"
            + str(
                self.model[index]
            )
        )
        
    return ACTIONS[decision]


def find_objects(self, object_coordinates, current_pos, field, return_coords=False):
    x, y = current_pos
    if len(object_coordinates) == 0:
        if return_coords:
            return 0, -1, None
        return 0, -1
    else:
        try:
            min_distance_ind = np.argmin(
                np.sum(
                    np.abs(np.asarray(object_coordinates) - np.asarray(current_pos)),
                    axis=1,
                )
            )
        except Exception as e:
            print(np.asarray(object_coordinates))
            print(np.asarray(current_pos))
            raise e

        object_coord = object_coordinates[min_distance_ind]

        distance = np.linalg.norm(np.asarray(object_coord) - np.asarray(current_pos))
        neighbour_tiles = np.array(
            [
                [x-1, y] if field[x-1, y] == 0 else [1000, 1000],
                [x, y+1] if field[x, y+1] == 0 else [1000, 1000],
                [x+1, y + 1] if field[x+1, y] == 0 else [1000, 1000],
                [x, y-1] if field[x, y-1] == 0 else [1000, 1000],
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

    save_field = np.asarray(np.where(close_field == 0))

    best_next_spot = (-1, -1)
    min_length = 4
    for safe_spot in save_field.T:
        next_step, length = pathfinding(close_field, (3, 3), safe_spot)
        if length < min_length:
            min_length = length
            best_next_spot = next_step
    # 0"UP", 1"RIGHT", 2"DOWN", 3"LEFT"
    if min_length < 4 and min_length > 1:
        if best_next_spot == (3, 4):
            direction = 2
        elif best_next_spot == (3, 2):
            direction = 0
        elif best_next_spot == (2, 3):
            direction = 3
        elif best_next_spot == (4, 3):
            direction = 1

    elif min_length >= 4:
        direction = -2

    else:
        direction = -1

    return direction, min_length


def pathfinding(field, start, end):

    matrix = field[:, :]
    matrix = np.where(matrix == 1, -1, matrix)
    matrix = np.where(matrix == 0, 1, matrix)

    grid = Grid(matrix=matrix)
    start = grid.node(*start)
    end = grid.node(end[1], end[0])
    finder = AStarFinder()
    path, runs = finder.find_path(start, end, grid)
    if runs >= 4 or len(path) == 0:
        return (-1, -1), runs
    elif runs >= 2 and len(path) != 0:
        return path[1], runs

    else:
        return path[0], runs


def find_crates(self, field, position):

    field_array = np.asarray(field)
    indexes = np.where(field_array == 1)
    crate_coords = np.column_stack((indexes[0], indexes[1]))
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

    (b,a) = position
    bombs = [((y, x), t) for ((x, y), t) in game_state["bombs"]]
    
    bomb_coordinates = [(y, x) for ((x, y), t) in game_state["bombs"]]
    explosion_map = game_state["explosion_map"].T
 
    coins = [(y,x) for (x,y) in game_state["coins"]]

    # check in which directions walls are: UP-RIGHT-DOWN-LEFT

    

    # search for bombs, crates and space in surroundings
    field[a,b]=30
    #print(field)
    surroundings =[1 if (field[a-1,b]==-1 or field[a-1,b]==1) else 0,
                        1 if (field[a,b+1]==-1 or field[a,b+1]==1) else 0, 
                        1 if (field[a+1,b]==-1 or field[a+1,b]==1) else 0,
                        1 if (field[a,b-1]==-1 or field[a,b-1]==1) else 0]

    field = add_bomb_path_to_field(bombs, explosion_map, field)
    save_direction, save_distance = find_safe_spot(self, field, (a, b))

    # find coin target
    coin_distance, self.coin_direction, self.target = find_objects(
        self,
        coins,
        (
            a,
            b,
        ),
        field,
        return_coords=True,
    )

    # bombs_coordinates = [(x, y) for ((x, y), t) in game_state["bombs"]]
    bomb_distance, self.bomb_direction = find_objects(
        self,
        bomb_coordinates,
        (
            a,
            b,
        ),
        field,
    )
    crate_coords = find_crates(self, field, (a, b))

    crate_distance, crate_direction = find_objects(
        self,
        crate_coords,
        (
            a,
            b,
        ),
        field,
    )
    
    if self.coin_direction == -1:
        features = np.array(
            surroundings
            + [
                crate_direction,
                self.bool_bomb,
                save_direction,
            ]
        )
    else:
        features = np.array(
            surroundings
            + [
                self.coin_direction,
                self.bool_bomb,
                save_direction,
            ]
        )
    self.logger.info(str(features))
    return str(features)


def feature_index(self, features):
    index = self.feature_dict[features]
    return index
