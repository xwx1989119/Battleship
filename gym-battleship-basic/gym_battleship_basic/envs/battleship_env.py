import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

DEFAULT_FLEET = [('Carrier', 5),
                 ('Battleship', 4),
                 ('Cruiser', 3),
                 ('Submarine', 3),
                 ('Destroyer', 2)]


class BattleshipEnv(gym.Env):
    def __init__(self, board_shape=(10,10), verbose=False, obs_3d=False):
        self.shape = board_shape
        self.board = Board(board_shape, DEFAULT_FLEET)
        n_action = self.board.shape[0] * self.board.shape[1]
        self.action_space = spaces.Discrete(n_action)
        self.np_random, self.seed = seeding.np_random()
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=self.board.shape,
                                            dtype=np.int8)
        self.obs_3d = obs_3d
        if self.obs_3d:
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=self.board.shape+(3,),
                                                dtype=np.int8)
        else:
            self.observation_space = spaces.Box(low=0, high=2,
                                                shape=self.board.shape,
                                                dtype=np.int8)
        self.state = 0
        self.verbose = verbose

    def step(self, shot):
        done = False
        y, x = divmod(shot, self.board.shape[0])
        if self.verbose:
            print('current shot {} with x {} y {}'.format(shot, x, y))
        reward = -1
        reduent_shot = False
        if self.board.shots[y, x] == 1:
            reward -= 10
            if self.verbose:
                print('duplicate shot found with action', shot)
            reduent_shot = True
        self.board.shots[y, x] = 1
        # ship_map = np.nan_to_num(self.board.map / self.board.map)
        ship_map = np.where(self.board.map <= 0, self.board.map, 1)
        observation = self.board.shots + (self.board.shots * ship_map)
        if (observation[y, x] == 2) and (not reduent_shot):
            reward += 10
            ship_id = self.board.map[y, x]
            # print(ship_id)
            # print(type(self.board.ships))
            # print(type(self.board.ships[ship_id]))
            self.board.ships[ship_id].health -= 1
            if self.verbose:
                print('ship hit ', ship_id, 'remaining health', self.board.ships[ship_id].health)
            if self.board.ships[ship_id].health == 0:
                self.board.sunk += [ship_id]
                if self.verbose:
                    print('ship sunk: ', self.board.sunk)
            done = True if len(self.board.sunk) == len(self.board.ships) else False
        if self.obs_3d:
            obs_3d = np.zeros(self.shape + (3,))
            obs_3d[..., 0] = np.where(observation > 0, 1, 0)
            obs_3d[..., 1] = np.where(observation == 1, 1, 0)
            obs_3d[..., 2] = np.where(observation == 2, 1, 0)
            return obs_3d, reward, done, {}
        else:
            return observation, reward, done, {}

    def render(self, mode='human'):
        # TO-DO - implement rendering for console
        return NotImplemented

    def reset(self):
        self.board = Board(self.shape, DEFAULT_FLEET)  # reset the board and ship locations
        if self.obs_3d:
            return np.zeros(self.shape + (3,))
        else:
            return self.board.shots


class Ship(object):
    def __init__(self, name, length, ship_id):
        self.name = name
        self.sunk = []
        self.id = ship_id
        self.length = length
        self.health = length
        self.loc = None

    def placement(self, loc):
        self.loc = loc


class Board(object):
    def __init__(self, board_shape, fleet):
        self.shape = board_shape
        # self.fleet = fleet
        self.ships = {}
        for i in range(len(fleet)):
            self.ships[i+1] = Ship(fleet[i][0], fleet[i][1], i+1)
        # map to mark those cell has been occupied
        self.shots = np.zeros(board_shape)
        self.map = np.zeros([self.shape[0], self.shape[1]], dtype=np.int16)
        self.ship_placement()
        self.sunk = []

    def ship_placement(self):
        for ship_id, ship in self.ships.items():
            while True:
                pos = np.random.randint(self.shape[0] * self.shape[1])
                direction = np.random.randint(2)
                y, x = divmod(pos, self.shape[0])
                if direction == 0:
                    ship_loc = self.map[y, x:x+ship.length]
                else:
                    ship_loc = self.map[y:y+ship.length, x]
                u, c = np.unique(ship_loc, return_counts=True)
                if u[0] == 0 and c[0] == ship.length:
                    if direction == 0:
                        self.map[y, x:x + ship.length] = ship.id
                        ship.placement([(y, x),(y, x + ship.length)])
                    else:
                        self.map[y:y + ship.length, x] = ship.id
                        ship.placement([(y, x),(y + ship.length, x)])
                    break