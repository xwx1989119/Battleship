import gym
import pickle
from keras.models import load_model
from agents.CNN_Agent import CNN_Agent, revoloving_train


with open(r'.\..\data\huntSearch_agentGames.pickle', 'rb') as handle:
    sample_data = pickle.load(handle)
env = gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False, obs_3d=True)
print('finished load inputs')
toy = False
trained_model = load_model(r'.\..\agents\CNN_AgentModel_v2.model')
if toy:
    quick_test_data = dict((k, sample_data[k]) for k in [1, 2, 3]
                           if k in sample_data)
    trained_agent = revoloving_train(quick_test_data, env, 2, 100)
else:
    trained_agent = revoloving_train(sample_data, env, 1000, 1000)
    print('saving models')