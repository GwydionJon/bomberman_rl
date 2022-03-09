from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, feature_index, act
import random as random
import numpy as np
import json


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
STATE_FEATURES = 2
# parameters
alpha = 0.01  # learning rate
gamma = 0.6  # discout factor


# additional events:
CHASING_COIN = "CHASING_COIN"
MOVED_TOWARDS_BOMB = "MOVED_TOWARDS_BOMB"
MOVED_FROM_BOMB = "MOVED_FROM_BOMB"
GOOD_BOMB_PLACEMENT = "GOOD_BOMB_PLACEMENT"
BAD_BOMB_PLACEMENT = "BAD_BOMB_PLACEMENT"


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

    reward = reward_from_events(self, events, self_action)

    # Implementing SARSA method

    old_action_value = self.model[old_index, ACTIONS.index(self_action)]

    new_action_value = ACTIONS.index(act(self, new_game_state))
    new_val = old_action_value + alpha * (
        reward + gamma * self.model[new_index, new_action_value] - old_action_value
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

    if self_action is None or old_game_state is None or new_game_state is None:
        return

    events = add_own_events(self, events, self_action, old_game_state)

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
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )

    train_model(self, last_game_state, last_action, events)

    with open("model_dict.json", "w") as fp:
        json.dump(self.feature_dict, fp)

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str], self_action) -> int:
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


    """
    game_rewards = {
        e.COIN_COLLECTED: 3,
        e.KILLED_OPPONENT: 5,
        e.COIN_FOUND: 1,
        e.WAITED: -2,
        e.INVALID_ACTION: -10,
        e.CRATE_DESTROYED: 0.3,
        e.KILLED_SELF: -15,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        CHASING_COIN: 3,
        MOVED_TOWARDS_BOMB: -5,
        MOVED_FROM_BOMB: 5,
        GOOD_BOMB_PLACEMENT: 2,
        BAD_BOMB_PLACEMENT: -2,
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


def add_own_events(self, events, action, game_state):

    # bomb next to crate
    if self.next_to_box and ACTIONS.index(action) == 5:
        events.append(GOOD_BOMB_PLACEMENT)

    # chek if at an intersection:
    intersection = 0
    for i in [7, 11, 13, 17]:
        if self.surroundings[i] != 1:
            intersection += 1
    # don't blow up bomb if no creates are nearby
    if intersection == 4 and ACTIONS.index(action) == 5:
        events.append(BAD_BOMB_PLACEMENT)

    if len(self.distance_trace) > 2:
        self.logger.debug(
            f"distance trace: {self.distance_trace[-1]}, {self.distance_trace[-2]}"
        )

        if self.distance_trace[-1] < self.distance_trace[-2]:
            events.append(CHASING_COIN)

    if self.bomb_direction != -1 and not ACTIONS.index(action) == 5:

        if ACTIONS.index(action) == self.bomb_direction:
            events.append(MOVED_TOWARDS_BOMB)

        elif ACTIONS.index(action) != self.bomb_direction:
            events.append(MOVED_FROM_BOMB)
    return events
