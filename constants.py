

# SIZE OF THE GAME
BOARD_SIZE = 7                        # board size (less complexity)
NUM_WALLS = 3                           # number of walls each player starts with


# PROGRAM PURPOSE
'''If you want a human to vs a trained agent, then turn these on
    Note: These can be toggles on or off during the games'''

DISPLAY_GAME = True                     # false to avoid pygame altogether
INITIALLY_HUMAN_PLAYING = False         # ultimate test of intelligence
INITIALLY_USING_ONLY_INFERENCE = False
RESTORE = False                         # signifies if agent should be loaded from tensorflow checkpoint (from disc)

INITIAL_GAME_DELAY = 0                  # initial value for game_delay, which simply slows down the game so we can watch the agents plays ;)
GAME_DELAY_SEONDS = 1                   # if game_delay is switched on, this is the delay used


# TRAINING PARAMETERS
NUM_GAMES = 100

REWARD_WIN = 1.0                        # big bucks
REWARD_BEING_ALIVE = -.04               # yikes

MEMORY_SIZE = 2000                     # max number of (s,a,s',r) samples to store for learning at once
BATCH_SIZE = 100                         # how many actions from memory to learn from at a time

MOVE_ACTION_PROBABILITY = .80           # training wheels to encorage the agents to move more often
GAMMA = 0.80                            # future reward discount factor (bellman equation)

# how often to take random actions for the sake of exploration
# decays over games starts from 1 and goes to 0 asymptotically
STARTING_EXPLORATION_PROBABILITY = 1.0
ENDING_EXPLORATION_PROBABILITY = 0.0
EXPLORATION_PROBABILITY_DECAY = 0.00005 # decay of epsilon

PRINT_UPDATE_FREQUENCY = 10


# DISPLAY PARAMETERS
SCREEN_SIZE = 200
SQUARE_TO_WALL_SIZE_RATIO = 5
AGENT_COLOR_TOP = (230, 46, 0) # red
AGENT_COLOR_BOT = (0, 0, 255) # blue
WALL_COLOR = (255, 255, 255) # white
SQUARE_COLOR = (0, 0, 0) # black
SCREEN_BG = (32, 32, 32) #grey

class BoardElement():
    """ constants that define what the board can hold and what the NN sees as inputs """
    EMPTY = 0           # NN is fed this as input for empty grid spaces
    WALL = -2           # same here for walls
    SELF_AGENT = 10      # nn input
    ENEMY_AGENT = 2     # nn input
    AGENT_TOP = "T"
    AGENT_BOT = "B"
    WALL_HORIZONTAL = "H"
    WALL_VERTICAL = "V"