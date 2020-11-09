import gym
import gym_battleship_basic
import numpy as np
import pickle
from tqdm import tqdm
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
import time


class CNN_Agent(object):
    def __init__(self, env=None, trained_model=None, log_saving=False):
        self.log_saving = log_saving
        self.obs_hist = None
        self.model_estimates = None
        self.shot_log = []
        if env is None:
            self.env = gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False)
        else:
            self.env = env
        if trained_model is not None:
            self.model = trained_model
        else:
            if self.env.obs_3d:
                self.model = self.define_model(self.env.observation_space.shape)
            else:
                self.model = self.define_model(self.env.observation_space.shape +(1,))

    def reset_log(self):
        self.obs_hist = None
        self.model_estimates = None
        self.shot_log = []

    def define_model(self, input_shape):
        X_input = Input(input_shape)
        X = ZeroPadding2D((3, 3))(X_input)
        X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
        X = BatchNormalization(axis=3, name='bn0')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((2, 2), name='max_pool')(X)
        X = Flatten()(X)
        X = Dense(100, activation='sigmoid', name='fc')(X)
        model = Model(inputs=X_input, outputs=X, name='HappyModel')
        model.compile(loss='binary_cross_entropy', optimizer=Adam(lr=1e-3))
        return model

    def train(self, train_x, train_y, batch_size, epochs):
        self.model.fit(train_x, train_y,
                       batch_size=batch_size,
                       epochs=epochs, verbose=2)

    def act(self, obs):
        act_values = self.model.predict(obs.reshape((1,) + self.env.observation_space.shape))[0]
        if self.log_saving:
            if self.model_estimates is None:
                self.model_estimates = act_values.reshape((1,) + act_values.shape)
            else:
                self.model_estimates = np.concatenate([self.model_estimates, act_values.reshape((1,) + act_values.shape)])
        if len(obs.shape) == 3:
            obs = obs[..., 0] + obs[..., 2]
        flatten_obs = obs.flatten()
        act_values[flatten_obs > 0] = -1  # assign shoot position as -1
        return np.argmax(act_values)

    def test(self, env):
        obs, done, ep_reward = env.reset(), False, 0
        i = 0
        while not done:
            i += 1
            action = self.act(obs)
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


def train_data_process(data_dict, verbose=False):
    game_list = data_dict.keys()
    # need to set the output shape first
    tot_rounds = 0
    for i in game_list:
        tot_rounds += data_dict[i]['observations'].shape[0]
    obs_shape = data_dict[game_list[0]]['observations'].shape[1:]
    data_x = np.zeros((tot_rounds,)+obs_shape)
    data_y = np.zeros((tot_rounds, 100))
    x_obs_start = 0
    for i in tqdm(game_list):
        start_time = time.time()
        rounds = data_dict[i]['observations'].shape[0]
        test_x = data_dict[i]['observations']
        test_y = data_dict[i]['target']
        time_stamp1 = time.time()
        x_process_time = start_time - time_stamp1
        test_y = np.where(test_y > 0, 1, 0)
        time_stamp2a = time.time()
        test_y = np.repeat(test_y.reshape(1, len(test_y)), rounds, axis=0)
        time_stamp2b = time.time()
        y_process_time1 = time_stamp2a - time_stamp1
        y_process_time2 = time_stamp2b - time_stamp2a
        data_x[x_obs_start: x_obs_start + rounds, :, :, :] = test_x
        data_y[x_obs_start: x_obs_start + rounds, :] = test_y
        x_obs_start += rounds
        time_stamp3 = time.time()
        concat_time = time_stamp3 - time_stamp2b
        if verbose:
            print('for game {} x process time {}; y process time {} and {}; concat_time {}'.format(i, x_process_time,
                                                                                                   y_process_time1,
                                                                                                   y_process_time2,
                                                                                                   concat_time))
    return data_x, np.array(data_y)


def test_play(agent, env, num_games):
    step_record = []
    reward_record = []
    data_dict = {}
    for i in tqdm(range(num_games), position=0, leave=True):
        total_step, reward = agent.test(env)
        step_record += [total_step]
        reward_record += [reward]
        data_dict[i] = {'observations': agent.obs_hist, 'target': env.board.map.flatten()}
    return data_dict, step_record, reward_record


def revoloving_train(starting_data, env, round=20, games_per_round = 1000):
    print('starting processing data')
    train_x, train_y = train_data_process(starting_data)
    CNN_Agent1 = CNN_Agent(env=env,log_saving=True)
    print('starting initial model fitting')
    CNN_Agent1.train(train_x, train_y,128,3)
    for i in tqdm(range(round)):
        print('training round {}, gameplay starts'.format(i))
        new_data_dict, step_record, reward_dict =\
            test_play(CNN_Agent1, env, games_per_round)
        print('training round {}, data process starts'.format(i))
        train_x, train_y = train_data_process(new_data_dict)
        CNN_Agent1.train(train_x, train_y,
                         batch_size=128,
                         epochs=3)
        if i % 25 == 0:
            CNN_Agent1.model.save('CNN_AgentModel.model')
    return CNN_Agent1


if __name__ == '__main__':
    with open(r'.\..\data\huntSearch_agentGames.pickle', 'rb') as handle:
        sample_data = pickle.load(handle)
    env = gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False, obs_3d=True)
    print('finished load inputs')
    toy = False
    trained_model = load_model(r'.\..\agents\CNN_AgentModel_v2.model')
    if toy:
        quick_test_data = dict((k, sample_data[k]) for k in [1,2,3]
                                        if k in sample_data)
        # x, y = train_data_process(quick_test_data, True)
        trained_agent = revoloving_train(quick_test_data, env, 2, 100)
    else:
        # load_model = load_model('CNN_AgentModel.model')
        trained_agent = revoloving_train(sample_data, env, 1000, 1000)
        print('saving models')
    #
    # with open(r'.\CNN_Agent.pickle', 'wb') as f:
    #     pickle.dump(trained_agent, f, protocol=pickle.HIGHEST_PROTOCOL)

