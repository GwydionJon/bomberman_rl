from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import (
    state_to_features,
    feature_index,
    act,
    add_bomb_path_to_field,
    find_crates,
)
import random as random
import numpy as np
import json
import pandas as pd
import os


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
STATE_FEATURES = 2
# parameters
alpha = 0.01  # learning rate
gamma = 0.3  # discout factor


# additional events:
CHASING_COIN = "CHASING_COIN"
MOVED_TOWARDS_BOMB = "MOVED_TOWARDS_BOMB"
MOVED_FROM_BOMB = "MOVED_FROM_BOMB"
GOOD_BOMB_PLACEMENT = "GOOD_BOMB_PLACEMENT"
BAD_BOMB_PLACEMENT = "BAD_BOMB_PLACEMENT"
MOVED_IN_DANGER = "MOVED_IN_DANGER"
MOVED_FROM_DANGER = "MOVED_FROM_DANGER"
CHASING_CRATE = "CHASING_CRATE"


def create_model(self):
    print("create model in train.py")
    q_table = np.zeros(
        (STATE_FEATURES, len(ACTIONS))
    )  # features times number of actions
    return q_table


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # (s, a, r, s')

    if self.first_training_round is True:
        self.model = create_model(self)  # =q_table
    else:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def train_model(self, old_game_state, self_action, events, new_game_state=None):

    # get row of model
    old_index = feature_index(self, state_to_features(self, old_game_state))

    while len(self.model) <= len(self.feature_dict):
        self.model = np.vstack([self.model, np.zeros(len(ACTIONS))])

    if new_game_state is not None:
        new_index = feature_index(self, state_to_features(self, new_game_state))
    else:
        # this should hopefully allow training at the last turn.
        # if not the model can't ever learn that it blew itself up.
        new_index = old_index

    # parameters

    reward = reward_from_events(self, events)

    # Implementing SARSA method
    epsilon = 0.1

    old_action_value = self.model[old_index, ACTIONS.index(self_action)]

    if np.random.rand() < (1 - epsilon):
        new_action = np.argmax(self.model[new_index])

    else:
        new_action = ACTIONS.index(
            np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        )

    new_val = old_action_value + alpha * (
        reward + gamma * self.model[new_index, new_action] - old_action_value
    )

    self.model[old_index, ACTIONS.index(self_action)] = new_val


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )

    if self_action is None:
        return

    if old_game_state is None:
        old_game_state = new_game_state

    if new_game_state is None:
        new_game_state = old_game_state

    events = add_own_events(self, events, self_action, old_game_state, new_game_state)

    train_model(self, old_game_state, self_action, events, new_game_state)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    events = add_own_events(self, events, last_action, last_game_state, last_game_state)
    train_model(self, last_game_state, last_action, events, last_game_state)

    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )

    self.distance_trace = []
    with open("model_dict.json", "w") as fp:
        json.dump(self.feature_dict, fp)

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    save_game_statistic(self, last_game_state)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.

    Current own Events:
    CHASING_COIN = "CHASING_COIN"
    MOVED_TOWARDS_BOMB = "MOVED_TOWARDS_BOMB"
    MOVED_FROM_BOMB = "MOVED_FROM_BOMB"
    GOOD_BOMB_PLACEMENT = "GOOD_BOMB_PLACEMENT"
    BAD_BOMB_PLACEMENT = "BAD_BOMB_PLACEMENT"
    MOVED_IN_DANGER = "MOVED_IN_DANGER"
    MOVED_FROM_DANGER = "MOVED_FROM_DANGER"
    CHASING_CRATE = "CHASING_CRATE"
    """
    game_rewards = {
        e.COIN_COLLECTED: 15,
        e.KILLED_OPPONENT: 5,
        e.COIN_FOUND: 1,
        e.WAITED: -9,
        e.INVALID_ACTION: -20,
        e.CRATE_DESTROYED: 0.3,
        e.KILLED_SELF: -40,
        e.MOVED_DOWN: -3,
        e.MOVED_LEFT: -3,
        e.MOVED_RIGHT: -3,
        e.MOVED_UP: -3,
        CHASING_COIN: 5,
        CHASING_CRATE: 3,
        MOVED_TOWARDS_BOMB: -5,
        MOVED_FROM_BOMB: 5,
        GOOD_BOMB_PLACEMENT: 7,
        BAD_BOMB_PLACEMENT: -20,
        MOVED_IN_DANGER: -5,
        MOVED_FROM_DANGER: 5,
    }
    reward_sum = 0
    reward_events = []

    for event in events:
        if event in game_rewards:
            reward_events.append(event)
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(reward_events)}")
    self.logger.info("")
    return reward_sum


def add_own_events(self, events, action, old_game_state, new_game_state):

    old_player_coor = old_game_state["self"][3]
    new_player_coor = new_game_state["self"][3]

    coin_coordinates = old_game_state["coins"]
    if len(coin_coordinates) != 0:
        old_coin_distances = np.linalg.norm(
            np.subtract(coin_coordinates, old_player_coor), axis=1
        )
        new_coin_distances = np.linalg.norm(
            np.subtract(coin_coordinates, new_player_coor), axis=1
        )

        if min(new_coin_distances) < min(old_coin_distances):

            events.append(CHASING_COIN)

    # reward moving from bomb
    bombs = old_game_state["bombs"]
    bomb_coordinates = [(x, y) for ((x, y), t) in bombs]

    if len(bomb_coordinates) != 0:
        old_bomb_distance = np.linalg.norm(
            np.subtract(bomb_coordinates, old_player_coor), axis=1
        )
        new_bomb_distance = np.linalg.norm(
            np.subtract(bomb_coordinates, new_player_coor), axis=1
        )

        if min(new_bomb_distance) < min(old_bomb_distance):

            events.append(MOVED_TOWARDS_BOMB)

        else:
            events.append(MOVED_FROM_BOMB)

    # penelize standing in danger.

    old_field = add_bomb_path_to_field(
        old_game_state["bombs"],
        old_game_state["explosion_map"],
        old_game_state["field"].T,
    )
    new_field = add_bomb_path_to_field(
        new_game_state["bombs"],
        new_game_state["explosion_map"],
        new_game_state["field"].T,
    )

    if old_field[old_player_coor] == 2 and new_field[new_player_coor] == 0:
        events.append(MOVED_FROM_DANGER)
    if old_field[old_player_coor] == 0 and new_field[new_player_coor] == 2:
        events.append(MOVED_IN_DANGER)

    # find boxes in nearest vicinty

    # chasing crate
    crates = find_crates(self, old_field, old_player_coor)

    if len(crates) != 0:
        old_crate_distances = np.linalg.norm(
            np.subtract(crates, old_player_coor), axis=1
        )
        new_crate_distances = np.linalg.norm(
            np.subtract(crates, new_player_coor), axis=1
        )

        if min(new_crate_distances) < min(old_crate_distances):

            events.append(CHASING_CRATE)

    surroundings = []
    x, y = old_player_coor
    for i in range(-1, 2):
        if (x + i) < old_field.shape[0]:
            surroundings.append(old_field[x + i, y])

        if (y + i) < old_field.shape[1]:
            surroundings.append(old_field[x, y + i])
        else:
            surroundings.append(-1)
    # bomb next to crate
    if 1 in surroundings and ACTIONS.index(action) == 5:
        events.append(GOOD_BOMB_PLACEMENT)

    # don't blow up bomb if no creates are nearby
    if 1 not in surroundings and ACTIONS.index(action) == 5:
        events.append(BAD_BOMB_PLACEMENT)

    return events


def save_game_statistic(self, game_state):
    filename = "learning_stat.csv"

    round = game_state["round"]
    steps = game_state["step"]
    _, score, self.bool_bomb, (x, y) = game_state["self"]

    old_count = 0
    header = True
    if os.path.exists(filename):
        header = False
        df_old = pd.read_csv(filename)
        old_count = int(df_old["round"].values[-1])

    summary = {"round": [1 + old_count], "steps": [steps], "score": [score]}

    df = pd.DataFrame.from_dict(summary)

    df.to_csv(filename, mode="a", header=header)
