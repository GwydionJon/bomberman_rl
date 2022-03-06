from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, feature_index
import random as random
import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT' ]  #, 'BOMB']
STATE_FEATURES = 53 

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

def create_model(self):
    q_table= np.zeros((STATE_FEATURES,len(ACTIONS))) #features times number of actions
    return q_table



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.trace = [] 
    if self.first_training_round is True:
        self.model = create_model(self) # =q_table
    else: 
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)
    if self_action is None or old_game_state is None or new_game_state is None:
        return
    
    old_index = feature_index(state_to_features(self,old_game_state))
    new_index = feature_index(state_to_features(self, new_game_state))

    alpha= 0.1
    gamma= 0.5
    #epsilon = 0.9
    reward = reward_from_events(self,events)
    old_val = self.model[old_index,ACTIONS.index(self_action)]
    new_val = old_val + alpha*(reward + gamma*np.max(self.model[new_index]) - old_val)
    
    self.model[old_index,ACTIONS.index(self_action)] = new_val
    
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(self, old_game_state), self_action, state_to_features(self,new_game_state), reward_from_events(self, events)))

    _, score, bool_bomb, (x, y) = new_game_state['self']
    self.trace.append([x,y])
    #print("new game event")

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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))
    #print(self.model)
    # Store the model
    with open("my-saved-model_v2.pt", "wb") as file:
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
        e.WAITED: -0.3 ,
        e.INVALID_ACTION: -10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #if len(self.trace) >10:
      #  if self.trace[-1] == self.trace[-3]:
      #      reward_sum += -1
      #  elif self.trace[-1]==self.trace[-5]:
      #      reward_sum += -0.5
       # if np.linalg.norm(self.trace[-1] -self.target) < np.linalg.norm(self.trace[-2] - self.target):
       #     print("distance rduced")
      #      reward_sum += 2
     #   else:
     #       reward_sum += -5
    #print(reward_sum)
    #print(self.trace)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
