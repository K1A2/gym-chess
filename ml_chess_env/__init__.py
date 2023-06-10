from ml_chess_env.envs.chess_greedy import ChessGreedyEnv
from gym.envs.registration import register


register(
    id="ChessGreedyEnv",
    entry_point="ml_chess_env.envs:ChessGreedyEnv",
)

register(
    id="ChessGameEnv",
    entry_point="ml_chess_env.envs:ChessGameEnv",
)
