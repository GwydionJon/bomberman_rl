from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
import random as random
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten, InputLayer
from tensorflow.keras.optimizers import Adam
import pandas as pd

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
STATE_FEATURES = 27


def create_model(self):

    model = Sequential()
    model.add(InputLayer(input_shape=(STATE_FEATURES)))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(15, activation="relu"))

    model.add(Dense(len(ACTIONS), activation="linear"))

    model.compile(loss="mse", optimizer=Adam())
    model.build()
    print("done")
    return model


def align_target_model(model1, model2):
    # load model 1 onto model2
    model2.set_weights(model1.get_weights())


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # (s, a, r, s')
    if self.first_training_round == True:
        self.model = create_model(self)  # =q_table
        self.future_target = create_model(self)
        align_target_model(self.model, self.future_target)

        self.history = {}
        self.history["old_features"] = np.zeros((400, STATE_FEATURES))
        self.history["actions"] = np.zeros(400, dtype=int)
        self.history["rewards"] = np.zeros(400)
        self.history["new_features"] = np.zeros((400, STATE_FEATURES))


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

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)
    if self_action is None or old_game_state is None or new_game_state is None:
        return

    step = old_game_state["step"] - 1
    reward = reward_from_events(self, events, self_action)
    self.history["new_features"][step] = state_to_features(self, new_game_state)
    self.history["old_features"][step] = state_to_features(self, old_game_state)
    self.history["actions"][step] = int(ACTIONS.index(self_action))
    self.history["rewards"][step] = reward


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
    step = last_game_state["step"] - 1
    reward = reward_from_events(self, events, last_action)
    self.history["new_features"][step] = state_to_features(self, last_game_state)
    self.history["old_features"][step] = state_to_features(self, last_game_state)
    self.history["actions"][step] = int(ACTIONS.index(last_action))
    self.history["rewards"][step] = reward

    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )

    train_models(self)
    self.model.save("tensorflow_model")


def train_models(self):
    # parameters
    gamma = 0.1

    old_features, actions, rewards, new_features = self.history.values()

    # for i in range(len(old_features)):
    target = self.model.predict(old_features.reshape(-1, STATE_FEATURES))
    target_future = self.future_target.predict(new_features.reshape(-1, STATE_FEATURES))

    target[:, actions] = rewards + gamma * np.amax(target_future, axis=1)
    self.model.fit(
        old_features.reshape(-1, STATE_FEATURES),
        target.reshape(-1, len(ACTIONS)),
        epochs=1,
        verbose=0,
    )


def reward_from_events(self, events: List[str], self_action) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 5,
        e.COIN_FOUND: 1,
        e.WAITED: -2,
        e.INVALID_ACTION: -10,
        e.CRATE_DESTROYED: 0.3,
        e.KILLED_SELF: -50,
    }
    reward_sum = 0
    reward_events = []

    for event in events:
        if event in game_rewards:
            reward_events.append(event)
            reward_sum += game_rewards[event]

    # encurage blowing up crates
    if self.next_to_box and ACTIONS.index(self_action) == 5:
        reward_sum += 1
        reward_events.append("Bomb next to Crate")

    # chek if at an intersection:
    intersection = 0
    for i in [7, 11, 13, 17]:
        if self.surroundings[i] != 1:
            intersection += 1
    # don't blow up bomb if no creates are nearby
    if intersection == 4 and ACTIONS.index(self_action) == 5:
        reward_sum += -1
        reward_events.append("no crates")

    if self.coin_direction != -1:
        if ACTIONS.index(self_action) == self.coin_direction:
            reward_sum += 0.5
            reward_events.append("moved in direction of coin")

        elif ACTIONS.index(self_action) != self.coin_direction:
            reward_sum += -0.7
            reward_events.append("moved not in direction of coin")

    if self.bomb_direction != -1 and not ACTIONS.index(self_action) == 5:

        if ACTIONS.index(self_action) == self.bomb_direction:
            reward_sum += -4
            reward_events.append("moved in direction of bomb")

        elif ACTIONS.index(self_action) != self.bomb_direction:
            reward_sum += 4
            reward_events.append("moved not in direction of bomb")

    # if self.bool_bomb == False and e.WAITED in events:
    #     reward_sum -= 3

    # # encurage running towards coin
    # # if len(self.distance_trace) > 2 and not e.COIN_COLLECTED in events:
    # #     # the closer the higher the value
    # #     if self.distance_trace[-2] < self.distance_trace[-1]:
    # #         reward_sum += 0.6
    # #         reward_events.append("towards coin")

    # #     elif self.distance_trace[-2] > self.distance_trace[-1]:
    # #         reward_sum += -0.6
    # #         reward_events.append("away from coin")

    # # encurage running away from bomb
    # if len(self.bomb_trace) > 2:
    #     # the closer the higher the value
    #     if self.bomb_trace[-2] < self.bomb_trace[-1]:
    #          reward_sum += -3
    #          reward_events.append("towards bomb")

    #     elif self.bomb_trace[-2] > self.bomb_trace[-1]:
    #         reward_sum += 3
    #         reward_events.append("away from bomb")

    # punish bomb dropping when no bomb available
    # if e.BOMB_DROPPED in events and not self.bool_bomb:
    #     reward_sum += -1
    #     reward_events.append("no bomb available")

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(reward_events)}")
    self.logger.info("")
    return reward_sum
