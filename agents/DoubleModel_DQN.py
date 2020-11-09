import gym
import gym_battleship_basic
import os
import numpy as np
import random
from keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model
from tqdm import tqdm
from agents.util import train_data_process
import copy
import pickle
# reference: https://gist.github.com/yashpatel5400/049fe6f4372b16bab5d3dab36854f262#file-mountaincar-py


class DoubleModel_DQN:
    def __init__(self, env, model=None, log_saving=False):
        self.log_saving = log_saving
        self.env = env
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.7
        self.learning_rate = 0.0001
        self.exploration_min = 0.01
        self.exploration_rate = 0.4
        self.tau = .125
        self.obs_hist = None
        self.shot_log = []
        self.model_estimates = None
        self.state_shape = self.env.observation_space.shape
        # self.weight_backup = r'.\model\DQN_V3_06082020_3.model'
        # self.weight_backup = r'.\model\DQN_V3_06092020_1.model'
        self.weight_backup = r'.\model\DQN_V3_06102020_2.model'
        if model is not None:
            self.model = model
            self.target_model = model
        else:
            self.model = self.create_model(self.env.observation_space.shape)
            self.target_model = self.create_model(self.env.observation_space.shape)

    def reset_log(self):
        self.obs_hist = None
        self.model_estimates = None
        self.shot_log = []

    def create_model(self, input_shape):
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
        model = Model(inputs=X_input, outputs=X, name='DQN_Battleship')
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        print(model.summary())
        return model

    def model_pretrain(self, game_logs, batch_size, epochs):
        train_x, train_y = train_data_process(game_logs, reward=True)
        self.target_model.fit(train_x, train_y,
                              batch_size=batch_size,
                              epochs=epochs, verbose=2)
        self.save_model(self.weight_backup)
        self.model.load_weights(self.weight_backup) # assign same weights

    def act(self, state, explore=True):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if explore and (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        act_values = self.model.predict(state)[0]
        if self.log_saving:
            if self.model_estimates is None:
                self.model_estimates = act_values.reshape((1,) + act_values.shape)
            else:
                self.model_estimates = np.concatenate([self.model_estimates, act_values.reshape((1,) + act_values.shape)])
        act_values[state[..., 0].flatten() != 0] = -99
        return np.argmax(act_values)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 256
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        state_input = np.array([])
        target_input = np.array([])
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            if sum(state_input.shape) == 0:
                state_input = state
                target_input = target
            else:
                state_input = np.concatenate([state_input, state])
                target_input = np.concatenate([target_input, target])
        self.model.fit(state_input, target_input, batch_size=32, epochs=2, verbose=1)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.target_model.save(fn)

    def test(self, env):
        obs, done, ep_reward = env.reset(), False, 0
        i = 0
        while not done:
            i += 1
            obs = np.reshape(obs, (1,) + env.observation_space.shape)
            action = self.act(obs, explore=False)
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

    def train(self, env, trials=1000, game_logs=None):
        if game_logs is not None:
            self.model_pretrain(game_logs, 128, 4)
        steps = []
        for trial in range(trials):
            cur_state = env.reset()
            cur_state = np.reshape(cur_state, (1,) + env.observation_space.shape)
            total_reward = 0
            total_step = 0
            done = False
            while not done:
                action = self.act(cur_state)
                # print(action)
                new_state, reward, done, _ = env.step(action)
                new_state = np.reshape(new_state, (1,) + env.observation_space.shape)
                self.remember(cur_state, action, reward, new_state, done)
                cur_state = new_state
                total_reward += reward
                total_step += 1
            self.replay()  # internally iterates default (prediction) model
            if trial % 100 == 0:
                self.target_train()
                self.save_model(self.weight_backup)
            steps += [total_step]
            if len(steps) <= 200:
                mean_steps = np.mean(steps)
            else:
                mean_steps = np.mean(steps[-200:])
            print("Completed in {} trials with reward {} and step {}, average steps {}".format(trial,
                                                                                               total_reward, total_step,
                                                                                               mean_steps))
        self.save_model(self.weight_backup)


if __name__ == "__main__":
    test_env = gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False, obs_3d=True)
    with open(r'.\..\data\huntSearch_agentGames.pickle', 'rb') as handle:
        sample_data = pickle.load(handle)
    dqn_agent = DoubleModel_DQN(env=test_env, log_saving=False)
    dqn_agent.train(test_env, 30000, sample_data)
    # main(test_env)
