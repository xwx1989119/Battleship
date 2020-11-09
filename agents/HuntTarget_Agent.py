import numpy as np

class HuntTargetAgent(object):
    def __init__(self, board_shape, verbose=False, log_saving=False):
        self.verbose = verbose
        self.shape = board_shape
        self.log_saving = log_saving
        self.obs_hist = None
        self.strategy_mode = 'S'  # initialize to S as

        self.hunt_queue = []
        self.shot_log = []
        self.prev_obs = np.zeros([self.shape[0], self.shape[1]], dtype=np.int16)
        self.neighbour_moves = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    def reset_log(self):
        self.shot_log = []
        self.obs_hist = None

    def add_hunt_queue(self, obs):
        hit_info = obs - self.prev_obs
        hit_cell = np.where(hit_info > 1)
        hit_cell = list(zip(hit_cell[0], hit_cell[1]))
        if len(hit_cell) == 1:
            if self.verbose:
                print('there is hit start add hunt list')
            hit_cell = hit_cell[0]
            for move in self.neighbour_moves:
                new_cell = (hit_cell[0] + move[0], hit_cell[1] + move[1])
                if (new_cell[0] < self.shape[0]) and (new_cell[1] < self.shape[0]) and (new_cell[0] >= 0) \
                        and (new_cell[1] >= 0) and (obs[new_cell[0]][new_cell[1]] == 0) and (
                        new_cell not in self.hunt_queue):
                    self.hunt_queue.append(new_cell)
            if self.verbose:
                print('current queue is {}'.format(self.hunt_queue))

    def action(self, obs):
        if len(obs.shape) == 3:
            obs = obs[..., 0] + obs[..., 2]  # flatten the inputs
        self.add_hunt_queue(obs)
        if len(self.hunt_queue) > 0:
            self.strategy_mode = 'H'
            next_move = self.hunt_queue[0]
            self.hunt_queue = self.hunt_queue[1:]
            return next_move[0], next_move[1]
        else:
            self.strategy_mode = 'S'
            i, j = np.nonzero(np.nan_to_num(obs / obs) - 1)
            idx = np.random.choice(len(i), 1, replace=False)
            if self.verbose:
                print('remaining space', len(i))
            return i[idx[0]], j[idx[0]]

    def test(self, env):
        obs, done, ep_reward = env.reset(), False, 0
        i = 0
        while not done:
            i += 1
            y, x = self.action(obs)
            if len(obs.shape) == 3:
                obs = obs[..., 0] + obs[..., 2]  # flatten the inputs
            self.prev_obs = obs
            action = y * env.shape[0] + x
            obs, reward, done, _ = env.step(action)
            if self.log_saving:
                if i == 1:
                    self.obs_hist = obs.reshape((1,) + obs.shape)
                    self.shot_log = [action]
                else:
                    self.obs_hist = np.concatenate([self.obs_hist, obs.reshape((1,) + obs.shape)])
                    self.shot_log += [action]
            ep_reward += reward
        return i, ep_reward


