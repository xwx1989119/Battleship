import numpy as np
from tqdm import tqdm
import time
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import os


def train_data_process(data_dict, verbose=False, reward=False):
    game_list = data_dict.keys()
    # need to set the output shape first
    tot_rounds = 0
    for i in game_list:
        tot_rounds += data_dict[i]['observations'].shape[0]
    obs_shape = data_dict[i]['observations'].shape[1:]
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
        if reward:
            test_y = np.where(test_y>0, 9, -1)
        else:
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


def agent_test(agent, env, game_rounds, agent_name='battleship', model_agent=False, log_save=False):
    step_record = []
    reward_record = []
    data_dict = {}
    for i in tqdm(range(game_rounds)):
        try:
            if agent.log_saving:
                agent.reset_log()
            total_step, reward = agent.test(env)
            step_record += [total_step]
            reward_record += [reward]
            if log_save:
                data_dict[i] = {'observations': agent.obs_hist,
                                'target': env.board.map.flatten(),
                                'shot': agent.shot_log}
                if model_agent:
                    data_dict[i]['shot'] = agent.shot_log
                    data_dict[i]['model_estimates'] = agent.model_estimates
        except:
            pass
    result_df = pd.DataFrame({'AgentName': [agent_name] * len(step_record),
                              'steps': step_record,
                              'reward': reward_record})
    return result_df, data_dict


def single_frame_viz( frame_num,game_hist=None,  model_obj=None, obs_3d=False, model_agent=False,
                      board_shape=(10,10), value_range=(-2,2), title=None, label_c='black'):
    x_len = range(board_shape[0])
    y_len = range(board_shape[1])
    if model_obj is not None:
        shot = model_obj.shot_log[frame_num]
        obs = model_obj.obs_hist[frame_num]
        if model_agent:
            model_estimates = model_obj.model_estimates[frame_num].reshape(board_shape)
    elif game_hist is not None :
        shot =game_hist['shot'][frame_num]
        obs = game_hist['observations'][frame_num]
        if model_agent:
            model_estimates = game_hist['model_estimates'][frame_num].reshape(board_shape)
    else:
        raise ValueError

    i1,j1 = divmod(shot, board_shape[0])
    fig, ax = plt.subplots(figsize=(8,8))
    if obs_3d:
        curr_game = obs[...,1]*-2 +obs[...,2]*2
    else:
        curr_game = obs.reshape(board_shape)
        curr_game = np.where(curr_game ==1, -2, curr_game)
    if model_agent:
            # model_values = game_hist['model_estimates'][frame_num].reshape(board_shape)
        im = ax.imshow(model_estimates, cmap=plt.cm.seismic,vmin=value_range[0], vmax=value_range[1])
    else:
        im = ax.imshow(curr_game, cmap=plt.cm.seismic, vmin=value_range[0], vmax=value_range[1])
    for txt in ax.texts:
        txt.set_visible(False)
    for i in range(board_shape[1]):
        for j in range(board_shape[0]):
            if (i != i1) | (j != j1):
                if curr_game[i, j] == 2:
                    text = ax.text(j, i, 'Hit', ha="center", va="center", color=label_c)
                elif curr_game[i, j] == -2:
                    text = ax.text(j, i, 'Miss', ha="center", va="center", color=label_c)
    if curr_game[i1, j1] == -2:
        ax.text(j1,i1,  'Shot & Miss',ha="center", va="center", color=label_c)
    else:
        ax.text(j1,i1,  'Shot & Hit', ha="center", va="center", color=label_c)
    ax.set_xticks(np.arange(len(x_len)))
    ax.set_yticks(np.arange(len(y_len)))
    ax.set_xticklabels(x_len)
    ax.set_yticklabels(y_len)
    if title is not None:
        ax.set_title(title)
    return fig


def game_viz(board_shape, agent_name, game_hist, label_c='blue', anim_flag=False, obs_3d=False, model_agent=False,
             value_range=(-2, 2), output_path=r"./asset/image"):
    x_len = range(board_shape[0])
    y_len = range(board_shape[1])

    map_record = np.zeros(board_shape)
    map_record[...] = np.mean(value_range)  # assign to the middle value.
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(map_record, cmap=plt.cm.seismic, vmin=value_range[0], vmax=value_range[1])
    ax.set_xticks(np.arange(len(x_len)))
    ax.set_yticks(np.arange(len(y_len)))
    ax.set_xticklabels(x_len)
    ax.set_yticklabels(y_len)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(x_len)):
        for j in range(len(y_len)):
            text = ax.text(j, i, map_record[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Gameplay for {}".format(agent_name))
    fig.tight_layout()
    if anim_flag:
        def update(i):
            label = 'timestep {0}'.format(i)
            shot = game_hist['shot'][i]
            i1, j1 = divmod(shot, board_shape[0])
            if obs_3d:
                curr_game = game_hist['observations'][i][..., 1] * -2 + game_hist['observations'][i][..., 2] * 2
            else:
                curr_game = game_hist['observations'][i].reshape(board_shape)
                curr_game = np.where(curr_game == 1, -2, curr_game)

            if model_agent:
                model_values = game_hist['model_estimates'][i].reshape(board_shape)
                im = ax.imshow(model_values, cmap=plt.cm.seismic, vmin=value_range[0], vmax=value_range[1])
            else:
                im = ax.imshow(curr_game, cmap=plt.cm.seismic, vmin=value_range[0], vmax=value_range[1])
            for txt in ax.texts:
                txt.set_visible(False)
            for i in range(board_shape[1]):
                for j in range(board_shape[0]):
                    if (i != i1) | (j != j1):
                        if curr_game[i, j] == 2:
                            text = ax.text(j, i, 'Hit',
                                           ha="center", va="center", color=label_c)
                        elif curr_game[i, j] == -2:
                            text = ax.text(j, i, 'Miss',
                                           ha="center", va="center", color=label_c)
            if curr_game[i1, j1] == -2:
                ax.text(j1, i1, 'Shot & Miss',
                        ha="center", va="center", color="r")
            else:
                ax.text(j1, i1, 'Shot & Hit',
                        ha="center", va="center", color="r")

            return im, ax
        # np.arange(0, game_hist['observations'].shape[0])
        anim = FuncAnimation(fig, update, frames=game_hist['observations'].shape[0],
                             interval=400, repeat=True)
        anim.save(os.path.join(output_path, agent_name + '_Game.gif'), dpi=80, writer='imagemagick')
        return anim
    else:
        return fig



if __name__ == '__main__':
    with open(r'.\..\data\huntSearch_agentGames.pickle', 'rb') as handle:
        sample_data = pickle.load(handle)

    quick_test_data = dict((k, sample_data[k]) for k in [1, 2, 3]
                           if k in sample_data)
    train_x, train_y = train_data_process(quick_test_data, True, True)