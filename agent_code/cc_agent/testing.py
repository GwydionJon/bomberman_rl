import numpy as np


coins = np.array([[1, 3], [4, 5], [2, 3], [8, 1]])
free_tiles = np.array([[4, 5], [2, 3], [8, 1], [2, 1], [1, 0], [1, 2]])
x, y = [1, 1]
current = np.array([x, y])

# min_distance =np.argmin(np.sum(np.abs(coins-current),axis=1))
# target = coins[min_distance] #Setting coordinates of coin/target
# neighbour_tiles = np.array([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
# possible_steps = []
# for i in range(len(neighbour_tiles)):
#      if (neighbour_tiles[i] == free_tiles).all(1).any()==True:
#          possible_steps.append(neighbour_tiles[i])
#      else:
#      #very big value so it will not be picked, bbut such that the action order stays the same
#          possible_steps.append((100,100))
# new_pos = np.argmin(np.sum(np.abs(possible_steps - target),axis=1))


# Boardx, Boardy = np.meshgrid(np.linspace(0,16,17),np.linspace(0,16,17))
# Board = []
# for x in np.linspace(0,16,17):
#     for y in np.linspace(0,16,17):
#         Board.append([x,y])


Board = np.indices((17, 17)).transpose((1, 2, 0))
# free_tiles = Board[free_tiles]
# Finding minimum distance to a coin
min_distance = np.argmin(np.sum(np.abs(coins - current), axis=1))
target = np.array([coins[min_distance]])  # Setting coordinates of coin/target
print(target)
# array of possible movements - free tiles
neighbour_tiles = np.array([[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]])
possible_steps = np.zeros((4, 2))
for i in range(len(neighbour_tiles)):
    if (neighbour_tiles[i] == free_tiles).all(1).any() == True:
        possible_steps[i] = neighbour_tiles[i]
    else:
        # very big value so it will not be picked, bbut such that the action order stays the same
        possible_steps[i] = (100, 100)
print(possible_steps)
new_pos = np.argmin(np.sum(np.abs(possible_steps - target), axis=1))
field = np.array([[0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]])
find_walls = [
    1 if field[x + 1, y] == 1 else 0,
    1 if field[x, y + 1] == 1 else 0,
    1 if field[x - 1, y] == 1 else 0,
    1 if field[x, y - 1] == 1 else 0,
]


neighbour_tiles = np.array(
    [
        [x, y + 1] if field[x, y + 1] == 0 else [1000, 1000],
        [x + 1, y] if field[x + 1, y] == 0 else [1000, 1000],
        [x, y - 1] if field[x, y - 1] == 0 else [1000, 1000],
        [x - 1, y] if field[x - 1, y] == 0 else [1000, 1000],
    ]
)
possible_steps = np.zeros((4, 2))
# for i in range(len(neighbour_tiles)):
#     if (neighbour_tiles[i] == free_tiles).all(1).any()==True:
#         possible_steps[i] = neighbour_tiles[i]
#     else:
#    #very big value so it will not be picked, but such that the action order stays the same
#         possible_steps[i] = (1000,1000)
new_pos = np.argmin(np.sum(np.abs(neighbour_tiles - target), axis=1))


Q_table = np.random.randint(5, size=(10, 5))
