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
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import pandas as pd

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]  # , 'BOMB']
STATE_FEATURES = 7
batch_size = 40


def create_model(self):

    model = Sequential()
    model.add(Dense(STATE_FEATURES, activation="relu"))

    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(15, activation="relu"))

    model.add(Dense(len(ACTIONS), activation="linear"))

    model.compile(loss="mse", optimizer=Adam())
    model.build(input_shape=(STATE_FEATURES, 1))
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
    reward = reward_from_events(self, events)
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
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )

    reward = reward_from_events(self, events)

    train_models(self)


def train_models(self):
    # parameters
    gamma = 0.1

    print(self.history["new_features"][0])
    old_features, actions, rewards, new_features = self.history.values()

    for i in range(len(old_features)):
        target = np.mean(self.model.predict(old_features[i]), axis=0)
        target_future = np.mean(self.future_target.predict(new_features[i]), axis=0)
        print(target, actions[i], rewards[i], np.amax(target_future))
        target[actions[i]] = rewards[i] + gamma * np.amax(target_future)

        self.model.fit(old_features[i], target, epochs=1, verbose=0)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.WAITED: -0.3,
        e.INVALID_ACTION: -10,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    if len(self.distance_trace) > 3:
        # reward getting closer to the target coin
        if self.distance_trace[-1] < self.distance_trace[-2]:
            reward_sum += 1

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
