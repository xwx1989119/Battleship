import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod
import gym
import copy
from  gym_battleship_basic.envs.battleship_env import Board

DEFAULT_FLEET = [('Carrier', 5),
                 ('Battleship', 4),
                 ('Cruiser', 3),
                 ('Submarine', 3),
                 ('Destroyer', 2)]


class BattleshipGameMonteCarloTreeSearchNode(ABC):
    def __init__(self, state, game_board_shape=(10, 10), parent=None, max_expand=30, roll_out_max_depth=2,
                 roll_out_simulation_round=1, action=None, verbose=0):
        """
        :param state:  take env board as state which contains all info
        :param game_board_ship:  game maps
        :param parent:
        """
        self.action = action
        self.roll_out_max_depth = roll_out_max_depth
        self.max_expand = max_expand
        self.state = state
        self.parent = parent
        self._number_of_visits = 0.
        self._results = 0
        self.gamma = 0.95
        self._untried_actions = None
        self.game_board_shape = game_board_shape
        self.expand_count = 0
        self.children = []
        self.verbose =verbose
        self.roll_out_simulation_round = roll_out_simulation_round

    @staticmethod
    def place_ship_on_board(board, ship_id, ship_loc):
        if (ship_loc[0][0] == ship_loc[1][0]) and\
                (np.sum(board.map[ship_loc[0][0], ship_loc[0][1]:ship_loc[1][1]]) == 0):
            board.map[ship_loc[0][0], ship_loc[0][1]:ship_loc[1][1]] = ship_id
            board.ships[ship_id].loc = ship_loc
            return board
        elif (ship_loc[0][1] == ship_loc[1][1]) and\
                (np.sum(board.map[ship_loc[0][0]:ship_loc[1][0], ship_loc[0][1]]) == 0):
            board.map[ship_loc[0][0]:ship_loc[1][0], ship_loc[0][1]] = ship_id
            board.ships[ship_id].loc = ship_loc
            return board
        else:
            raise ValueError('Illegal ship location. please check ')

    @staticmethod
    def place_ship_on_board_maps(board, ship_id, ship_loc):
        if (ship_loc[0][0] == ship_loc[1][0]) and\
                (np.sum(board[ship_loc[0][0], ship_loc[0][1]:ship_loc[1][1]]) == 0):
            board[ship_loc[0][0], ship_loc[0][1]:ship_loc[1][1]] = ship_id
            return board
        elif (ship_loc[0][1] == ship_loc[1][1]) and\
                (np.sum(board[ship_loc[0][0]:ship_loc[1][0], ship_loc[0][1]]) == 0):
            board[ship_loc[0][0]:ship_loc[1][0], ship_loc[0][1]] = ship_id
            return board
        else:
            raise ValueError('illgeal ship location. pleace check ')

    def get_legal_actions(self, board):
        shot_maps = board.shots
        legal_action = []
        empty_cell = np.where(shot_maps == 0)
        for i in range(np.shape(empty_cell)[1]):
            legal_action += [empty_cell[1][i] + empty_cell[0][i] * self.game_board_shape[0]]
        return legal_action

    def is_fully_expanded(self):
        return (len(self.untried_actions) == 0) or (self.expand_count > self.max_expand)

    @property
    def untried_actions(self):
        empty_cell = np.where(self.state.shots == 0)
        if self._untried_actions is None:
            self._untried_actions = []
            for i in range(np.shape(empty_cell)[1]):
                self._untried_actions += [empty_cell[1][i] + empty_cell[0][i] * self.game_board_shape[0]]
        return self._untried_actions

    @property
    def n(self):
        return self._number_of_visits

    @property
    def q(self):
        return self._results

    def convert_loc(self, loc, board_shape):
        y, x = divmod(loc, board_shape[0])
        return y, x

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def is_terminal_node(self):
        """
        when all ships are sunk, game is done
        :return:
        """
        return len(self.state.sunk) == len(self.state.ships)

    def board_simulation(self):
        simu_board = copy.deepcopy(self.state)
        simu_board.map = np.zeros(simu_board.shape, dtype=np.int16)
        ship_map = np.where(self.state.map <= 0, self.state.map, 1)
        observation = self.state.shots + (self.state.shots * ship_map)
        avaiable_ship = []
        for ship_id in simu_board.ships.keys():
            if ship_id in simu_board.sunk:
                simu_board = self.place_ship_on_board(simu_board, ship_id, simu_board.ships[ship_id].loc)
            else:
                simu_board.ships[ship_id].health = simu_board.ships[ship_id].length
                simu_board.ships[ship_id].loc = None
                avaiable_ship += [ship_id]
        available_map = simu_board.map * simu_board.shots
        empty_cell = np.array(np.where(observation == 0)).T # all
        empty_cell = [tuple(item) for item in empty_cell]
        available_shot_cell = np.array(np.where(simu_board.map == 0)).T
        available_shot_cell = [tuple(item) for item in available_shot_cell if observation[tuple(item)] == 2]
        assigned_ship_dict = {}
        all_placed = False
        while not all_placed:
            _curr_legal_map = copy.deepcopy(available_map)
            _curr_empty_cell = copy.deepcopy(empty_cell)
            _curr_available_shot_cell = copy.deepcopy(available_shot_cell)
            assigned_ship_dict = {}
            for ship_id in avaiable_ship:
                curr_ship_len = simu_board.ships[ship_id].length
                if self.verbose == 2:
                    print('currently working on ship', ship_id)
                if len(_curr_available_shot_cell) > 0:
                    for _loc in _curr_available_shot_cell:
                        hrz_list = [((_loc[0], _loc[1] - shift), (_loc[0], _loc[1] - shift + curr_ship_len))
                                    for shift in range(curr_ship_len)
                                    if ((_loc[1] - shift) >= 0) &
                                    ((_loc[1] - shift + curr_ship_len) <= 10) & # cant go to next row
                                    (np.sum(_curr_legal_map[_loc[0],
                                            (_loc[1] - shift): (_loc[1] - shift + curr_ship_len)]) == 0)
                                    ]
                        vrt_list = [((_loc[0] - shift, _loc[1]), (_loc[0] - shift + curr_ship_len, _loc[1]))
                                    for shift in range(curr_ship_len)
                                    if ((_loc[0] - ship_id) >= 0) & ((_loc[0] - shift + curr_ship_len) <= 10) &
                                    (np.sum(_curr_legal_map[(_loc[0] - shift): (_loc[0] - shift + curr_ship_len),
                                            _loc[1]]) == 0)
                                    ]
                        all_loc_list = hrz_list + vrt_list
                        if len(all_loc_list) != 0:
                            picked_loc = np.random.choice(range(len(all_loc_list)))
                            picked_loc = all_loc_list[picked_loc]
                            if self.verbose == 2:
                                print('try to place ship ', ship_id, ' with location,', picked_loc)
                            _curr_legal_map = self.place_ship_on_board_maps(_curr_legal_map, 1, picked_loc)
                            assigned_ship_dict[ship_id] = picked_loc
                            _curr_available_shot_cell.remove(_loc)
                            break
                # if non of the shot cell satisfy current observations.
                if ship_id not in assigned_ship_dict.keys():
                    hrz_list = [((_loc[0], _loc[1]), (_loc[0], _loc[1] + curr_ship_len)) for _loc in empty_cell
                                if ((_loc[1] + curr_ship_len) <= 10) &  # cant go to next row
                                (np.sum(_curr_legal_map[_loc[0], (_loc[1]): (_loc[1] + curr_ship_len)]) == 0)
                                ]
                    vrt_list = [((_loc[0], _loc[1]), (_loc[0] + curr_ship_len, _loc[1]) )for _loc in empty_cell
                                if ((_loc[0] + curr_ship_len) <= 10) &  # cannt be out of the map
                                (np.sum(_curr_legal_map[_loc[0]: (_loc[0] + curr_ship_len), _loc[1]]) == 0)
                                ]
                    all_loc_list = hrz_list + vrt_list
                    if len(all_loc_list) == 0:
                        if self.verbose:
                            print('ship placement failed for ship {}'.format(str(ship_id)))
                        break
                    else:
                        picked_loc = np.random.choice(range(len(all_loc_list)))
                        picked_loc = all_loc_list[picked_loc]
                        if self.verbose == 2:
                            print('try to place ship ', ship_id, ' with location,', picked_loc)
                        # print('pre assign board:', _curr_legal_map)
                        _curr_legal_map = self.place_ship_on_board_maps(_curr_legal_map, 1, picked_loc)
                        assigned_ship_dict[ship_id] = picked_loc

            if len(assigned_ship_dict.keys()) == len(avaiable_ship):
                all_placed = True

        for _ship_id in assigned_ship_dict.keys():
            simu_board = self.place_ship_on_board(simu_board, _ship_id, assigned_ship_dict[_ship_id])

        return simu_board

    def move_on_board(self, action):
        """
        we first simulation a new board based on observation and then place our action
        :param action:
        :return:
        """
        new_board = self.board_simulation()
        done = False
        y, x = divmod(action, new_board.shape[0])
        if self.verbose == 2:
            print('current shot {} with x {} y {}'.format(action, x, y))
        reward = -1
        reduent_shot = False
        if new_board.shots[y, x] == 1:
            reward -= 10
            if self.verbose == 1:
                print('duplicate shot found with action', action)
            reduent_shot = True
        new_board.shots[y, x] = 1
        ship_map = np.where(new_board.map <= 0, new_board.map, 1)
        observation = new_board.shots + (new_board.shots * ship_map)
        if (observation[y, x] == 2) and (not reduent_shot):
            reward += 10
            ship_id = new_board.map[y, x]
            new_board.ships[ship_id].health -= 1
            if self.verbose == 2:
                print('ship hit ', ship_id, 'remaining health', new_board.ships[ship_id].health)
            if new_board.ships[ship_id].health == 0:
                new_board.sunk += [ship_id]
                if self.verbose:
                    print('ship sunk: ', new_board.sunk)
            done = True if len(new_board.sunk) == len(new_board.ships) else False
        return new_board, reward, done, {}

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        """
        our space is still large. we could limit our search to a smaller number.
        :return:
        """
        action = self.untried_actions.pop()
        if self.verbose == 2:
            print('##### Currently working on expanding the node ######')
        next_state, reward, done, _ = self.move_on_board(action)
        child_node = BattleshipGameMonteCarloTreeSearchNode(
            next_state, parent=self, action=action, verbose=self.verbose,
            roll_out_simulation_round=self.roll_out_simulation_round, max_expand=self.max_expand
        )
        self.expand_count += 1
        self.children.append(child_node)
        return child_node

    def rollout(self):
        current_rollout_state = copy.deepcopy(self.state)
        done = False
        result = 0
        rollout_depth = 0
        for i in range(self.roll_out_simulation_round):
            if self.verbose == 2:
                print('#########currently working on rollout simulation #', i)
            while (not done) and (rollout_depth <= self.roll_out_max_depth):
                if self.verbose == 2:
                    print('################  currently working on rollout simulation #', i, ' and depth:', rollout_depth)
                possible_moves = self.get_legal_actions(current_rollout_state)
                if len(possible_moves) == 0:
                    break
                action = self.rollout_policy(possible_moves)
                current_rollout_state, reward, done, _ = self.move_on_board(action)
                result += self.gamma ** i * reward
                rollout_depth += 1
        return result/self.roll_out_simulation_round

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results += result
        if self.parent:
            self.parent.backpropagate(result)