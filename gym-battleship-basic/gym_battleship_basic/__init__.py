from gym.envs.registration import register

register(
    id='battleshipBasic-v0',
    entry_point='gym_battleship_basic.envs:BattleshipEnv',
)