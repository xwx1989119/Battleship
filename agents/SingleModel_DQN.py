import gym
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym_battleship_basic
# Reference: https://github.com/GaetanJUVIN/Deep_QLearning_CartPole/blob/master/cartpole.py
# def _action_after_filter(state, action_values):
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model
from tqdm import tqdm
from agents.util import train_data_process
import pickle


class DQN_Agent(object):
    def __init__(self, state_size, action_size, verbose=False, model=None, log_saving=False):
        self.verbose = verbose
        self.weight_backup = r".\model\DQN_V2_06102020_2.model"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model_estimates = None
        self.learning_rate = 0.0001
        self.gamma = 0.1
        self.exploration_rate = 0.4
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.log_saving = log_saving
        self.obs_hist = None
        self.shot_log = []
        self.model_estimates = None
        if model is None:
            self.brain = self._build_model(self.state_size)
        else:
            self.brain = model

    def model_pretrain(self, game_logs, batch_size, epochs):
        train_x, train_y = train_data_process(game_logs, reward=True)
        self.brain.fit(train_x, train_y,
                       batch_size=batch_size,
                       epochs=epochs, verbose=2)
        self.save_model()

    def _build_model(self, input_shape):
        X_input = Input(input_shape)
        X = ZeroPadding2D((3, 3))(X_input)
        X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
        X = BatchNormalization(axis=3, name='bn0')(X)
        X = ZeroPadding2D((3, 3))(X)
        X = Conv2D(32, (7, 7), strides=(1, 1), name='conv1')(X)
        X = BatchNormalization(axis=3, name='bn1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), name='max_pool')(X)
        # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
        X = Flatten()(X)
        # X = Dense(100, activation='sigmoid', name='fc')(X)
        X = Dense(100, activation='linear', name='fc')(X)
        model = Model(inputs=X_input, outputs=X, name='HappyModel')
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        print(model.summary())
        return model

    def save_model(self):
        self.brain.save(self.weight_backup)

    def act(self, state, explore=True):
        if explore and (np.random.rand() <= self.exploration_rate):
            return random.randrange(self.action_size), 1
        act_values = self.brain.predict(state)[0]
        if self.log_saving:
            if self.model_estimates is None:
                self.model_estimates = act_values.reshape((1,) + act_values.shape)
            else:
                self.model_estimates = np.concatenate([self.model_estimates,
                                                       act_values.reshape((1,) + act_values.shape)])
        act_values[state[..., 0].flatten() != 0] = -99
        return np.argmax(act_values), 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        state_input = np.array([])
        target_input = np.array([])
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            if sum(state_input.shape) == 0:
                state_input = state
                target_input = target_f
            else:
                state_input = np.concatenate([state_input, state])
                target_input = np.concatenate([target_input, target_f])
        self.brain.fit(state_input, target_input, batch_size=32, epochs=2, verbose=1)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class BattleShipAgent:
    def __init__(self, agent=None, model=None, log_saving=False, verbose=False, episodes=10000):
        self.sample_batch_size = 256
        self.episodes = episodes
        self.env = gym.make('battleshipBasic-v0', board_shape=(10, 10),  verbose=False, obs_3d=True)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.verbose = verbose
        self.agent = agent
        self.obs_hist = None
        self.shot_log = []
        self.log_saving = log_saving
        self.model_estimates = None
        if self.agent is None:
            self.agent = DQN_Agent(state_size=self.state_size, action_size=self.action_size,
                                   model=model, log_saving=log_saving)

    def reset_log(self):
        self.obs_hist = None
        self.model_estimates = None
        self.shot_log = []
        self.agent.model_estimates = None

    def train(self):
        try:
            total_step_log = []
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, (1,) + self.state_size)
                done = False
                total_reward = 0
                total_step = 0
                explore_step_count = 0
                while not done:
                    action, explore_count = self.agent.act(state)
                    explore_step_count += explore_count
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, (1,)+ self.state_size)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    total_step += 1
                    if self.verbose:
                        print(reward)
                total_step_log += [total_step]
                print("Episode {}# Score: {} and step {} ({} explore steps), "
                      "current average steps {}".format(index_episode,
                                                        total_reward, total_step,
                                                        explore_step_count,
                                                        np.mean(total_step_log)))
                self.agent.replay(self.sample_batch_size)
                if (index_episode%100) == 0:
                    self.agent.save_model()
        finally:
            self.agent.save_model()

    def test(self, env):
        obs, done, ep_reward = env.reset(), False, 0
        i = 0
        explore_count = 0
        while not done:
            i += 1
            obs = np.reshape(obs, (1,) + self.state_size)
            action, _explore_count = self.agent.act(obs, explore=False)
            obs, reward, done, _ = env.step(action)
            if self.log_saving:
                if i == 1:
                    self.obs_hist = obs.reshape((1,) + obs.shape)
                    self.shot_log = [action]
                else:
                    self.obs_hist = np.concatenate([self.obs_hist, obs.reshape((1,) + obs.shape)])
                    self.shot_log += [action]
            ep_reward += reward
            explore_count += _explore_count
            # print(explore_count)
        self.model_estimates = self.agent.model_estimates
        return i, ep_reward


def test_n_rounds(env_test, agent, n_rounds):
    step_log = []
    reward_log = []
    for i in tqdm(range(n_rounds)):
        _step, _reward = agent.test(env_test)
        step_log += [_step]
        reward_log += [_reward]
    print('average steps {}'.format(np.mean(step_log)))


if __name__ == "__main__":
    test_flag = True
    Model_train = True
    # use_CNN_model = False
    # if use_CNN_model:
    #     trained_model = load_model(r'.\..\agents\CNN_AgentModel.model')
    #     battleship = BattleShipAgent(episodes=500, model=trained_model)
    # else:
    battleship = BattleShipAgent(episodes=1000)
    # test_n_rounds(env_test=gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False, obs_3d=True),
    #               agent=battleship,
    #               n_rounds=1
    #               )

    if Model_train:
        with open(r'.\..\data\huntSearch_agentGames.pickle', 'rb') as handle:
            sample_data = pickle.load(handle)
        battleship.agent.model_pretrain(sample_data, 256, 6)
        test_n_rounds(env_test=gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False, obs_3d=True),
                      agent=battleship,
                      n_rounds=1000
                      )
        battleship.train()
        test_n_rounds(env_test=gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False, obs_3d=True),
                      agent=battleship,
                      n_rounds=1000
                      )

    # if test_flag:
    #     env_test = gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False, obs_3d=True)
    #     step_log = []
    #     reward_log = []
    #     for i in tqdm(range(1000)):
    #         _step, _reward = battleship.test(env_test)
    #         step_log += [_step]
    #         reward_log += [_reward]
    #     print('average steps {}'.format(np.mean(step_log)))