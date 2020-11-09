import time
from agents.MCTS_Nodes import BattleshipGameMonteCarloTreeSearchNode
from agents.util import agent_test
import numpy as np
import copy
from tqdm import tqdm
# from multiprocessing import Pool
import functools
# https://github.com/int8/monte-carlo-tree-search


class MonteCarloTreeSearch(object):
    def __init__(self, node, verbose=1):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node
        self.verbose = verbose

    def best_action(self, simulations_number):
        """
        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action
        Returns
        -------
        """
        for _ in tqdm(range(0, simulations_number), position=0, leave=True):
            if self.verbose == 2:
                print('############## currently working on simulation, ', _)
                print(len(self.root._untried_actions))
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        if self.verbose == 1:
            print('root children #', len(self.root.children), 'visited time, ', self.root.n)
            children_visited = [[item.action, item.q, item.n] for item in self.root.children]
            print(children_visited)

        _action = self.root.best_child(c_param=0.).action
        # print('best action: ', _action)
        return self.root.best_child(c_param=0.).action

    def _tree_policy(self):
        """
        selects node to run rollout/playout for
        Returns
        -------
        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node


class MCTS_Battleship(object):
    def __init__(self, search_method=MonteCarloTreeSearch, roll_out_simulation_round=1, max_expand=60,
                 verbose=0, simulation_round=100, log_saving=False):
        self.search_method = search_method
        self.verbose = verbose
        self.simulation_round = simulation_round
        self.log_saving = log_saving
        self.roll_out_simulation_round = roll_out_simulation_round
        self.max_expand = max_expand

    def action(self, state):
        mcts = self.search_method(state, verbose=self.verbose)
        return mcts.best_action(self.simulation_round)

    def test(self, env):
        obs, done, ep_reward = env.reset(), False, 0
        i = 0
        while not done:
            i += 1
            env_board = copy.deepcopy(env.board)
            node_state = BattleshipGameMonteCarloTreeSearchNode(state=env_board, parent=None, verbose=self.verbose,
                                                                max_expand=self.max_expand,
                                                                roll_out_simulation_round=self.roll_out_simulation_round)
            if self.verbose >= 0:
                print('#' * 50)
                print('###### Round ', i, '############')
                print(obs)
                print('#' * 50)

            if len(node_state.untried_actions) <= 3:
                action = node_state.untried_actions[0]
            else:
                action = self.action(node_state)
            if len(obs.shape) == 3:
                obs = obs[..., 0] + obs[..., 2]  # flatten the inputs
            self.prev_obs = obs
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


if __name__ == '__main__':
    import gym
    import gym_battleship_basic
    env_mc = gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False, obs_3d=False)
    MCTS_Agent1 = MCTS_Battleship(verbose=1, simulation_round=100,
                                  roll_out_simulation_round=50, max_expand=100)
    import time
    start_time = time.time()
    tot_step, reward = MCTS_Agent1.test(env_mc)
    used_tiome = time.time() - start_time
    print('total step: {} total reward {} total used time {}'.format(str(tot_step), str(reward), str(used_tiome)))

    output_df, data_dict = agent_test(MCTS_Agent1, env_mc, game_rounds=2, agent_name='MCTS_battleship',
                                      model_agent=False, log_save=False)
    output_df.to_csv(r'MC_Agent_test.csv')