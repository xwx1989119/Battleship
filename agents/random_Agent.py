import numpy as np


class RandomAgent(object):
    def __init__(self, verbose=False, log_saving=False):
        self.verbose = verbose
        self.log_saving = log_saving
        self.obs_hist = None
        self.shot_log = []

    def action(self, obs):
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
            action = y * env.shape[0] + x
            obs, reward, done, _ = env.step(action)
            if self.log_saving:
                if i == 1:
                    self.obs_hist = obs.flatten()
                    self.shot_log = [action]
                else:
                    self.obs_hist = np.vstack([self.obs_hist, obs.flatten()])
                    self.shot_log += [action]
            ep_reward += reward
        return i, ep_reward

    def reset_log(self):
        self.obs_hist = None
        self.shot_log = []

