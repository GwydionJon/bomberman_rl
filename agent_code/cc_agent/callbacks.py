import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT' ]  #, 'BOMB']
STATE_FEATURES = 53 

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
    self.target = None
    
    if self.train and not os.path.isfile("my-saved-model_v2.pt"):
        print("First round")
        self.logger.info("Setting up model from scratch.")
        self.first_training_round = True
        self.model = np.zeros((STATE_FEATURES,len(ACTIONS)))
    elif self.train and os.path.isfile("my-saved-model_v2.pt"):
        print("second round")
        self.logger.info("Loading model from saved state.")
        self.first_training_round = False
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    else:
        self.first_training_round = False
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


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
    if self.first_training_round is True and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2])
    else:
        self.logger.debug("Querying model for action.")
        
        return ACTIONS[np.argmax(self.model[feature_index(state_to_features(self, game_state)),:])]

def find_coin(self, game_state: dict): #Function which finds nearest coin and decides then where to go
    _, score, bool_bomb, (x, y) = game_state['self']
    current = np.array([x,y]) # Curent position of agent
    coins = game_state['coins'] # Position of visible coins
    if len(coins)==0:
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2])
        return ACTIONS.index(action)
    #free_tiles = game_state['field'].T == 0 # Free tiles on field
    #Board = np.indices((17,17)).transpose((1,2,0))
    #free_tiles = Board[free_tiles]
    field = game_state['field'].T
    #Finding minimum distance to a coin
    
  #  self.last_coin_number = len(coins)
    
   # print(self.target)
    #if self.last_coin_number != len(coins):
     #   print("set target none")
      #  self.target == None
 
    # choose target
    #if self.target is None:
    min_distance =np.argmin(np.sum(np.abs(coins-current),axis=1)) 
    self.target = np.array(coins[min_distance]) #Setting coordinates of coin/target
        
   # array of possible movements - free tiles :UP - RIGHT - DOWN - LEFT
    neighbour_tiles = np.array([[x, y +1] if field[x,y+1]==0 else [1000,1000], 
                                [x +1, y] if field[x+1,y]==0 else [1000,1000], 
                                [x, y -1] if field[x, y-1]==0 else [1000,1000],
                                [x- 1, y] if field[x-1,y]==0 else [1000,1000]])
   
    new_pos = np.argmin(np.sum(np.abs(neighbour_tiles - self.target),axis=1))
    #new_pos = possible_steps[new_pos]
    return new_pos # new acceptable position -> go there

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
     
# =============================================================================
#     
#     # For example, you could construct several channels of equal shape, ...
#     channels = []
#     channels.append(...)
#     # concatenate them as a feature tensor (they must have the same shape), ...
#     stacked_channels = np.stack(channels)
#     # and return them as a vector
#     return stacked_channels.reshape(-1)
# 
# =============================================================================
    
    _, score, bool_bomb, (x, y) = game_state['self']
    coin_target = find_coin(self, game_state) 
    field = game_state['field'].T
    # check in which directions walls are: UP-RIGHT-DOWN-LEFT
    find_walls= [1 if field[x, y+1] == -1 else 0, 
                 1 if field[x+1, y] == -1 else 0,
                 1 if field[x, y-1] == -1 else 0, 
                 1 if field[x-1, y] == -1 else 0]
    features = np.array(find_walls + [coin_target]) 
    return  features

def feature_index(features):
    return 2**3*4*features[0]+2**2*4*features[1]+2*4*features[2]+4*features[3]+features[4]