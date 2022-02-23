import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

def find_coin(self,game_state: dict): #Function which finds nearest coin and decides then where to go
    _, score, bool_bomb, (x, y) = game_state['self']
    current = np.array([x,y]) # Curent position of agent
    coins = game_state['coins'] # Position of visible coins
    free_tiles = game_state['field'] == 0 # Free tiles on field
    #Finding minimum distance to a coin
    min_distance =np.argmin(np.sum(np.abs(coins-current),axis=1)) 
    target = coins[min_distance] #Setting coordinates of coin/target
    # array of possible movements
    possible_steps = np.array([[x,y+1],[x+1,y],[x,y-1],[x-1,y]]) #Up, Right, Down, Left
   
    new_pos = np.argmin(np.sum(np.abs(possible_steps - target),axis=1))
    next_step = possible_steps[new_pos]
    return new_pos # yes -> go there
   
        

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    #best_step = find_coin(self, game_state)
    

    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])


    self.logger.debug("Querying model for action.")
    _, score, bool_bomb, (x, y) = game_state['self']
    #next_tile = find_coin(self, game_state) # np.random.choice(ACTIONS, p=self.model)
    # if x==next_tile[0] and y==next_tile[1]+1:
    #     Action = ACTIONS[0]
    # if x==next_tile[0]+1 and y==next_tile[1]:
    #     Action = ACTIONS[1]
    # if x==next_tile[0] and y==next_tile[1]-1:
    #     Action = ACTIONS[2]
    # else:
    #     Action = ACTIONS[3]
    return ACTIONS[find_coin(self, game_state)]

def state_to_features(game_state: dict) -> np.array:
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

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
