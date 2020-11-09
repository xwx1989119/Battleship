import numpy as np
import copy
from multiprocessing import Pool
from tqdm import tqdm
import functools
import pandas as pd
import time
from agents.util import agent_test
DEFAULT_FLEET = [('Carrier', 5),
                 ('Battleship', 4),
                 ('Cruiser', 3),
                 ('Submarine', 3),
                 ('Destroyer', 2)]

class MCAgent(object):
    def __init__(self, board_shape, multiprocess=True, verbose=False, log_saving=False, simulation_round=100):
        self.verbose = verbose
        self.shape = board_shape
        self.log_saving = log_saving
        self.obs_hist = None
        self.pool = Pool() if multiprocess else None
        self.simulation_round = simulation_round
        self.multiprocess = multiprocess
        self.hunt_queue = []
        self.shot_log = []
        self.prev_obs = np.zeros([self.shape[0], self.shape[1]], dtype=np.int16)
        self.model_estimates = None
        # self.pool = Pool()  # Create a multiprocessing Pool

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    @staticmethod
    def check_place(ship_loc, miss_map, curr_ship_map):
        if ship_loc[0][0] == ship_loc[1][0]:  # this is a horizontal ship
            miss_map_vals = miss_map[ship_loc[0][0], ship_loc[0][1]:ship_loc[1][1]]
            curr_ship_map_vals = curr_ship_map[ship_loc[0][0], ship_loc[0][1]:ship_loc[1][1]]
        elif ship_loc[0][1] == ship_loc[1][1]:
            miss_map_vals = miss_map[ship_loc[0][0]:ship_loc[1][0], ship_loc[0][1]]
            curr_ship_map_vals = curr_ship_map[ship_loc[0][0]:ship_loc[1][0], ship_loc[0][1]]
        else:
            return False
        miss_unique = np.unique(miss_map_vals)
        curr_unique = np.unique(curr_ship_map_vals)
        if (miss_unique[0] == 0) & (curr_unique[0] == 0) & (len(miss_unique) == 1) & (len(curr_unique) == 1):
            return True
        else:
            return False

    @staticmethod
    def assign_ship(target_baord, loc):
        if loc[0][0] == loc[1][0]:
            target_baord[loc[0][0], loc[0][1]:loc[1][1]] = 1
        else:
            target_baord[loc[0][0]:loc[1][0], loc[0][1]] = 1
        return target_baord

    @staticmethod
    def check_board(board, obs):
        sunk_ids = board.sunk
        simu_starting_board = np.zeros(board.shape, dtype=np.int16)
        curr_missing_board = np.zeros(board.shape, dtype=np.int16)
        occupied_cell = []
        missed_shot = np.where(obs == 1)
        missed_shot_loc = []
        for i in range(np.shape(missed_shot)[1]):
            missed_shot_loc += [missed_shot[1][i] + missed_shot[0][i] * 10]
        curr_missing_board[missed_shot] = 1
        hit_shot = np.where(obs == 2)
        hit_shot_loc = []
        for i in range(np.shape(hit_shot)[1]):
            hit_shot_loc += [hit_shot[1][i] + hit_shot[0][i] * 10]
        for sunk_id in sunk_ids:
            _sunk_locs = board.ships[sunk_id].loc
            if _sunk_locs[0][0] == _sunk_locs[1][0]:
                simu_starting_board[_sunk_locs[0][0], _sunk_locs[0][1]:_sunk_locs[1][1]] = 1
                curr_ship_cell = [_sunk_locs[0][0] * board.shape[0] + i for i in range(_sunk_locs[0][1], _sunk_locs[1][1])]
                assert len(curr_ship_cell) == board.ships[sunk_id].length
                occupied_cell += curr_ship_cell
            else:
                simu_starting_board[_sunk_locs[0][0]:_sunk_locs[1][0], _sunk_locs[0][1]] = 1
                curr_ship_cell = [i * board.shape[0] + _sunk_locs[0][1] for i in range(_sunk_locs[0][0], _sunk_locs[1][0])]
                assert len(curr_ship_cell) == board.ships[sunk_id].length
                occupied_cell += curr_ship_cell
        available_hit_cell = list(set(hit_shot_loc) - set(occupied_cell))
        empty_cell = list(set(np.arange(board.shape[0]*board.shape[1])) - set(occupied_cell) -
                          set(missed_shot_loc) - set(available_hit_cell))
        return simu_starting_board, curr_missing_board, empty_cell, available_hit_cell

    def convert_loc(self, loc, board_shape):
        y, x = divmod(loc, board_shape[0])
        return y, x

    def multiprocessing_simulation(self,  board, obs, simulation_num=1000):
        start_time = time.time()
        available_ship_ids = [_id for _id in board.ships.keys() if _id not in board.sunk]
        get_ship_time = time.time()
        if self.verbose:
            print('time used for get ship id, ', (get_ship_time - start_time))
        starting_board, missing_board, all_loc_sapce, hit_loc_space = \
            self.check_board(board, obs)
        check_board_time = time.time()
        if self.verbose:
            print('time used for check_board, ', (check_board_time - get_ship_time))
        ship_len_dict = {_id: board.ships[_id].length for _id in board.ships.keys()}
        partial_func = functools.partial(self.signle_simulation,
                                         available_ship_ids,
                                         starting_board,
                                         missing_board,
                                         all_loc_sapce,
                                         hit_loc_space,
                                         ship_len_dict)
        partial_func_time = time.time()
        if self.verbose:
            print('time used for partial function, ', (partial_func_time - check_board_time))
        pool_creation = time.time()
        if self.verbose:
            print('time used for pool function, ', (pool_creation - partial_func_time))
        if simulation_num == 1:
            all_output = partial_func(1)
            return all_output
        elif self.multiprocess:
            all_output = self.pool.map(partial_func, range(simulation_num))  # process data_inputs iterable with pool
        else:
            all_output = np.zeros((simulation_num,)+board.shape)
            for i in range(simulation_num):
                all_output[i] = partial_func(1)
        simulation_time = time.time()
        if self.verbose:
            print('time used for simulation function, ', (simulation_time - pool_creation))
        return np.sum(all_output, axis=0)

    def signle_simulation(self, available_ship_ids, starting_board,
                          missing_board, all_loc_sapce, hit_loc_space,ship_len_dict,count):
        map_finish = False
        board_shape = starting_board.shape
        tot_assign = 0
        illgeal_pos = missing_board + starting_board
        while not map_finish:
            curr_ship_ids = copy.copy(available_ship_ids)
            curr_simulation_board = copy.copy(starting_board)
            curr_all_loc_sapce = copy.copy(all_loc_sapce)
            curr_hit_loc_space = copy.copy(hit_loc_space)
            np.random.shuffle(curr_ship_ids)
            np.random.shuffle(curr_all_loc_sapce)
            assign_ships = []
            if len(hit_loc_space) > 0:
                np.random.shuffle(curr_hit_loc_space)
            for j in range(len(curr_ship_ids)):
                curr_ship_len = ship_len_dict[curr_ship_ids[j]]
                assigned = False
                # while not assigned
                if len(curr_hit_loc_space) > 0:
                    for k in range(len(curr_hit_loc_space)):
                        direction = np.random.randint(2)
                        if direction == 0: # horizontal
                            choice_list = [curr_hit_loc_space[k] - shift for shift in range(curr_ship_len)
                                           if ((curr_hit_loc_space[k] - shift) >= 0) &
                                           (curr_hit_loc_space[k] % 10 - shift + curr_ship_len <= 10)]
                            if len(choice_list) == 0:
                                break
                            loc = np.random.choice(choice_list, 1)[0]
                            ship_loc = [self.convert_loc(loc,board_shape),
                                        (self.convert_loc(loc,board_shape)[0],
                                         self.convert_loc(loc,board_shape)[1] + curr_ship_len)]
                        else: # vertical
                            choice_list = [curr_hit_loc_space[k] - shift * board_shape[0]
                                           for shift in range(curr_ship_len)
                                           if ((curr_hit_loc_space[k] - shift * board_shape[0]) >= 0) &
                                           ((curr_hit_loc_space[k] // 10 + (curr_ship_len - shift)) <= 10) &
                                           ((curr_hit_loc_space[k] // 10 + (curr_ship_len - shift)) >= 0)]
                            if len(choice_list) == 0:
                                break
                            loc = np.random.choice(choice_list, 1)[0]
                            ship_loc = [self.convert_loc(loc, board_shape),
                                        (self.convert_loc(loc,board_shape)[0] + curr_ship_len,
                                         self.convert_loc(loc, board_shape)[1])]
                        if self.check_place(ship_loc, missing_board, curr_simulation_board):
                            curr_simulation_board = self.assign_ship(curr_simulation_board, ship_loc)
                            ship_cells = [(ship_loc[0][0], ship_loc[0][1] + i) if direction == 0
                                          else (ship_loc[0][0] + i, ship_loc[0][1]) for i in range(curr_ship_len)]
                            for cell in ship_cells:
                                cell_num = cell[0] * board_shape[0] + cell[1]
                                if cell_num in curr_hit_loc_space:
                                    curr_hit_loc_space.remove(cell_num)
                            assign_ships += [curr_ship_ids[j]]
                            tot_assign += 1
                            assigned = True
                            break
                if not assigned:
                    for k in range(len(curr_all_loc_sapce)):
                        hrz_list =[_loc for _loc in curr_all_loc_sapce if(_loc % 10 + curr_ship_len <= 10) &
                                   (np.sum(illgeal_pos[self.convert_loc(_loc,board_shape)[0],
                                           self.convert_loc(_loc, board_shape)[1]:
                                           self.convert_loc(_loc, board_shape)[1] + curr_ship_len]) == 0)]
                        vert_list = [_loc for _loc in curr_all_loc_sapce if ((_loc // 10 + curr_ship_len) <= 10) &
                                     (np.sum(illgeal_pos[self.convert_loc(_loc, board_shape)[0]:
                                                         self.convert_loc(_loc,board_shape)[0] + curr_ship_len,
                                             self.convert_loc(_loc, board_shape)[1]]) == 0)]
                        if len(hrz_list) == len(vert_list) == 0:
                            break
                        elif len(hrz_list) == 0:
                            direction = 1
                        elif len(vert_list) == 0:
                            direction = 0
                        else:
                            direction = np.random.randint(2)

                        if direction == 0:
                            loc = np.random.choice(hrz_list, 1)[0]
                            ship_loc = [self.convert_loc(loc,board_shape),
                                        (self.convert_loc(loc,board_shape)[0],
                                         self.convert_loc(loc,board_shape)[1] + curr_ship_len)]
                        else:
                            loc = np.random.choice(vert_list, 1)[0]
                            ship_loc = [self.convert_loc(loc, board_shape),
                                        (self.convert_loc(loc,board_shape)[0] + curr_ship_len,
                                         self.convert_loc(loc, board_shape)[1])]
                        if self.check_place(ship_loc, missing_board, curr_simulation_board):
                            curr_simulation_board = self.assign_ship(curr_simulation_board, ship_loc)
                            curr_all_loc_sapce.remove(loc)
                            assign_ships += [curr_ship_ids[j]]
                            assigned = True
                            tot_assign += 1
                            break
                    if not assigned:
                        break  # ship assignment failed start-over again.
            if len(assign_ships) != len(curr_ship_ids):
                continue
            else:
                return curr_simulation_board

    def reset_log(self):
        self.shot_log = []
        self.obs_hist = None
        self.model_estimates = None

    def action(self, obs, board):
        if len(obs.shape) == 3:
            obs = obs[..., 0] + obs[..., 2]  # flatten the inputs
        mc_estimation_est_orig = self.multiprocessing_simulation(board, obs, self.simulation_round)
        mc_estimation_est = (mc_estimation_est_orig/ self.simulation_round).flatten()
        if self.log_saving:
            if self.model_estimates is None:
                self.model_estimates = mc_estimation_est.reshape((1,) + mc_estimation_est.shape)
            else:
                self.model_estimates = np.concatenate([self.model_estimates, mc_estimation_est.reshape((1,) + mc_estimation_est.shape)])
        flatten_obs = obs.flatten()
        mc_estimation_est[flatten_obs > 0] = -1  # assign shoot position as -1
        action_val = np.argmax(mc_estimation_est)
        if self.verbose:
            print(mc_estimation_est_orig)
            print(obs)
            print(mc_estimation_est)
            print(action_val)
        return action_val

    def test(self, env):
        obs, done, ep_reward = env.reset(), False, 0
        i = 0
        while not done:
            i += 1
            if self.verbose:
                print('###### Round ', i, '############')
                print(time.time())
            action = self.action(obs, env.board)
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
    pass
    # import gym
    # import gym_battleship_basic
    # env_mc = gym.make('battleshipBasic-v0', board_shape=(10, 10), verbose=False, obs_3d=False)
    # MC_Agent1 = MCAgent(board_shape=(10, 10),verbose=True, log_saving=True, simulation_round=1)
    # import time
    # start_time = time.time()
    # tot_step, reward = MC_Agent1.test(env_mc)
    # used_tiome = time.time() - start_time
    # print('game finished with time', used_tiome)
    # print('test Done')
    # MC_Agent1 = MCAgent(board_shape=(10, 10),multiprocess=True, verbose=False,  log_saving=True, simulation_round=500)
    # output_df, data_dict = agent_test(MC_Agent1, env_mc, game_rounds=1, agent_name='battleship',
    #                                   model_agent=False, log_save=False)
    # import pickle
    # with open (r'c:\temp\MC_AgentGameLog.pickle','wb') as f:
    #     pickle.dump(data_dict, f)
    # output_df.to_csv(r'MC_Agent_test.csv')

