import gym
import gym_battleship_basic
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

def test(dqn_model, test_env, log_saving=False):
    obs_hist = None
    obs, done, ep_reward = test_env.reset(), False, 0
    i = 0
    while not done:
        i += 1
        action = dqn_model.forward(obs)
        obs, reward, done,_ = env.step(action)
        if log_saving:
            if i == 1:
                obs_hist = obs.flatten()
            else:
                obs_hist = np.vstack([obs_hist, obs.flatten()])
        ep_reward += reward
    return i, ep_reward, obs_hist


def model_define(env):
    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model


if __name__ == '__main__':
    env = gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False, obs_3d=True)
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    model = model_define(env)
    dqn = DQNAgent(model=model,
                   nb_actions= env.action_space.n,
                   memory=memory,
                   nb_steps_warmup=10,
                   target_model_update=1e-2,
                   policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)
