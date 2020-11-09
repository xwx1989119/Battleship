# Battleship
## Introduction
This repo contains the battleship environment as well as 6 types of agents for this game. 

* __Random Agent__: agent that place shots randomly on the map
* __Hunt and Target Agent__: agent that first randomly search the map, and focus on the neighbour of hit cell when hit a target. 
* __CNN Agent__: use CNN model to predict the location of ships, and place shot on the cell with highest probability . 
* __Single and Double DQN Agent__: standard DQN agent that works on the Q-functions
* __Monte Carlo Agent__: use Monte Carlo simluation to estimate the probability of ship locations, then use the same strategy as CNN Agent 
* __Monte Carlo Tree Search__: use MCTS for this game.

Some quick notes:

* the code is not fully cleaned and optimized. For example, my MCTS and MC agent use the same logic to generate simluations. However, I used a newer version of simulation generation for MCTS. This could be confusing.
* MCTS agent is probably only a starting point for this method. I have built the major components but haven't got the time to further test and optimize it.

## Installation
First clone this repo and install the environment.
```sh
~ git clone https://github.com/M0r13n/battleships.git
~ pip install -e .\gym-battlehip-basic
```

required packages are:

* tqdm==4.39.0
* gym==0.17.3
* matplotlib==2.2.2
* Keras==2.2.4
* keras-rl==0.4.2
* pandas==0.23.3
* numpy==1.14.5


## repo structure

* \agent: agent code
* \agent\model:  CNN & DQN models
* \asset\data: game results, game logs for training, note current files are not the most updated data for model training. The trianing data is too big to be included. 
* \gym-battleship-basic: game environment

## Agent List

### Random Agent
![Random Agent Game Play](/asset/images/RandomAgent_Game.gif)


### Hunt and Target
![SegmentLocal](/asset/images/hunt_target_agent_Game2.gif)

### CNN
![SegmentLocal](/asset/images/CNN_Model_Agent_Game.gif)

### DQN 
![SegmentLocal](/asset/images/DQN_DoubleModel_Agent_Game.gif)


### MC  
![SegmentLocal](/asset/images/MC_Agent_Game.gif)
