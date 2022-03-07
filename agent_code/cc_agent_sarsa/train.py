from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, feature_index
import random as random
import numpy as np
import json


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]  # , 'BOMB']
STATE_FEATURES = 2


def create_model(self):
    print("create model in train.py")
    q_table = np.zeros((STATE_FEATURES, len(ACTIONS)))  # features times number of actions
    return q_table


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # (s, a, r, s')
    self.model = create_model(self)  # =q_table
    self.trace = []


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

    #Starting SARSA learning
    if self_action is None or old_game_state is None or new_game_state is None:
        return

    old_index = feature_index(self, state_to_features(self, old_game_state))

    # print(len(self.model) , len(self.feature_dict))
    while len(self.model) <= len(self.feature_dict):

        self.model = np.vstack([self.model, np.zeros(len(ACTIONS))])

        # self.feature_dict[new_hash] = len(self.feature_dict)

        # self.model.append(np.zeros(len(ACTIONS)))

    new_index = feature_index(self, state_to_features(self, new_game_state))

    alpha = 0.01
    gamma = 0.1
    epsilon = 0.1
    reward = reward_from_events(self, events)
    old_val = self.model[old_index, ACTIONS.index(self_action)]
    # Implementing SARSA method
    if np.random.rand() < (1- epsilon):
        new_action = np.argmax(self.model[new_index])
    
    else:
        new_action = ACTIONS.index(np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.2]))
    new_val = old_val + alpha * (
        reward + gamma * self.model[new_index,new_action] - old_val)

    self.model[old_index, ACTIONS.index(self_action)] = new_val

   # _, score, bool_bomb, (x, y) = new_game_state["self"]
    # self.trace.append([x,y])


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
    # print(self.model)
    # Store the model

    with open("model_dict.json", "w") as fp:
        json.dump(self.feature_dict, fp)

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


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
