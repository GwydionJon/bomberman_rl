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
    find_safe_spot,
)
import random as random
import numpy as np
import json
import pandas as pd
import os


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
STATE_FEATURES = 2
# parameters


# additional events:
CHASING_COIN = "CHASING_COIN"
MOVED_TOWARDS_BOMB = "MOVED_TOWARDS_BOMB"
MOVED_FROM_BOMB = "MOVED_FROM_BOMB"
GOOD_BOMB_PLACEMENT = "GOOD_BOMB_PLACEMENT"
BAD_BOMB_PLACEMENT = "BAD_BOMB_PLACEMENT"
MOVED_IN_DANGER = "MOVED_IN_DANGER"
MOVED_FROM_DANGER = "MOVED_FROM_DANGER"
CHASING_CRATE = "CHASING_CRATE"
CHASING_COIN_INSTEAD_OF_CRATE = "CHASING_COIN_INSTEAD_OF_CRATE"
CHASING_CRATE_INSTEAD_OF_COIN = "CHASING_CRATE_INSTEAD_OF_COIN"


def create_model(self):
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
    epsilon = self.reward_dict["epsilon"]
    alpha = self.reward_dict["alpha"]  # learning rate
    gamma = self.reward_dict["gamma"]  # discout factor

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

    """
    game_rewards = self.reward_dict
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

    old_field = old_game_state["field"].T
    new_field = new_game_state["field"].T

    new_bombs = [((y, x), t) for ((x, y), t) in new_game_state["bombs"]]
    old_bombs = [((y, x), t) for ((x, y), t) in old_game_state["bombs"]]

    new_bomb_coordinates = [(y, x) for ((x, y), t) in new_game_state["bombs"]]
    old_bomb_coordinates = [(y, x) for ((x, y), t) in old_game_state["bombs"]]

    new_explosion_map = new_game_state["explosion_map"].T
    old_explosion_map = old_game_state["explosion_map"].T

    new_coins = np.asarray(new_game_state["coins"]).T
    old_coins = np.asarray(old_game_state["coins"]).T

    old_field = add_bomb_path_to_field(
        old_bombs,
        old_explosion_map,
        old_field,
    )
    new_field = add_bomb_path_to_field(
        new_bombs,
        new_explosion_map,
        new_field,
    )

    old_player_coor = np.asarray(old_game_state["self"][3]).T
    # old_player_coor[0][1] = old_player_coor[1][0]
    new_player_coor = np.asarray(new_game_state["self"][3]).T

    old_save_direction, old_save_distance = find_safe_spot(
        self, old_field, (old_player_coor[0], old_player_coor[1])
    )
    new_save_direction, new_save_distance = find_safe_spot(
        self, new_field, (new_player_coor[0], new_player_coor[1])
    )
    # reward chasing coin
    if len(old_coins) != 0:
        try:
            old_coin_distances = np.linalg.norm(
                np.subtract(old_coins, old_player_coor), axis=1
            )
        except Exception as e:
            old_coin_distances = np.linalg.norm(
                np.subtract(np.asarray(old_coins).T, old_player_coor), axis=1
            )
        try:
            new_coin_distances = np.linalg.norm(
                np.subtract(np.asarray(old_coins).T, new_player_coor), axis=1
            )

        except Exception as e:
            print(old_coins)
            print(new_player_coor)
            raise e

        if min(new_coin_distances) < min(old_coin_distances):

            events.append(CHASING_COIN)

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

    # reward chasing coin instead of crate
    if len(old_coins) != 0 and len(crates) != 0:
        if min(new_crate_distances) >= min(old_crate_distances) and min(
            new_coin_distances
        ) < min(old_coin_distances):
            events.append(CHASING_COIN_INSTEAD_OF_CRATE)

        elif min(new_crate_distances) < min(old_crate_distances) and min(
            new_coin_distances
        ) >= min(old_coin_distances):
            events.append(CHASING_CRATE_INSTEAD_OF_COIN)

    # reward moving from bomb

    if len(old_bomb_coordinates) != 0:
        old_bomb_distance = np.linalg.norm(
            np.subtract(old_bomb_coordinates, old_player_coor), axis=1
        )
        new_bomb_distance = np.linalg.norm(
            np.subtract(old_bomb_coordinates, new_player_coor), axis=1
        )

        if min(new_bomb_distance) < min(old_bomb_distance):

            events.append(MOVED_TOWARDS_BOMB)

        else:
            if "BOMB_EXPLODED" not in events:
                events.append(MOVED_FROM_BOMB)

    # penelize standing in danger.

    if (
        old_field[old_player_coor[0], old_player_coor[1]] == 2
        and new_save_distance < old_save_distance
    ):
        events.append(MOVED_FROM_DANGER)
    if (
        old_field[old_player_coor[0], old_player_coor[1]] == 0
        and new_save_distance > old_save_distance
    ):
        events.append(MOVED_IN_DANGER)

    # find boxes in nearest vicinty

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
    if (1 in surroundings and old_save_distance <= 3) and ACTIONS.index(action) == 5:
        events.append(GOOD_BOMB_PLACEMENT)

    # don't blow up bomb if no creates are nearby
    if (1 not in surroundings or old_save_distance >= 4) and ACTIONS.index(action) == 5:
        events.append(BAD_BOMB_PLACEMENT)

    return events


def save_game_statistic(self, game_state):
    filename = "stat.json"

    round = game_state["round"]
    steps = game_state["step"]
    _, score, self.bool_bomb, (x, y) = game_state["self"]

    self.score.append(score)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            stats_dict = json.load(f)
    else:
        stats_dict = {}

    stats_dict.setdefault(str(list(self.reward_dict.values())), np.mean(self.score))
    stats_dict[str(list(self.reward_dict.values()))] = np.mean(self.score)

    with open(filename, "w") as fp:
        json.dump(stats_dict, fp)

    # old_count = 0
    # header = True
    # if os.path.exists("train_result.csv"):
    #     header = False
    #     df_old = pd.read_csv("train_result.csv")
    #     old_count = int(df_old["round"].values[-1])

    # summary = {"round": [1 + old_count], "steps": [steps], "score": [score]}

    # df = pd.DataFrame.from_dict(summary)

    # df.to_csv("train_result.csv", mode="a", header=header)
