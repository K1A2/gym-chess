import chess
import chess.svg
import numpy as np

import gym
from gym import spaces


# upper piece: white piece
# lower piece: black piece

piece_mapper = {
    'r': -1,
    'n': -2,
    'b': -3,
    'k': -4,
    'q': -5,
    'p': -6,
    'R': 1,
    'N': 2,
    'B': 3,
    'K': 4,
    'Q': 5,
    'P': 6,
    '.': 0
}


piece_values = {
    'r': -5,
    'n': -3,
    'b': -3,
    'k': -1,
    'q': -8,
    'p': -1,
    'R': 5,
    'N': 3,
    'B': 3,
    'K': 1,
    'Q': 8,
    'P': 1
}


class ChessGreedyEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=-6, high=6, shape=(64,), dtype=np.int32)  # required
        self.action_space = spaces.MultiDiscrete(4096)  # required

        self._auxiliary_reward_factor = 1
        self.win_reward = 1000
        self.lose_reward = -1000
        self.draw_reward = 0.0

    def _get_obs(self):
        # return flattened board array
        obs = np.array(list(map(piece_mapper.get, str(self._board).split())), dtype=np.int32)
        return obs

    def _get_info(self):
        self._action_mask = np.zeros(shape=(4096,), dtype=np.bool8)
        self._legal_moves = list(self._board.legal_moves)
        self._action_id_dict = {}

        for move in self._legal_moves:
            if move.promotion is not None and move.promotion != 5:  # promotion only for queen
                continue

            action_id = move.from_square * 64 + move.to_square
            self._action_mask[action_id] = True
            self._action_id_dict[action_id] = move
        
        material_value = self._get_material_value()
        result = self._board.result()
        if self._board.outcome() is not None:
            termination = str(self._board.outcome().termination)
        else:
            termination = ''

        return {'action_mask': self._action_mask, 'material_value': material_value,
                'result': result, 'termination': termination}
    
    def reset(self, seed=None, options=None):
        self._board = chess.Board()

        return self._get_obs(), self._get_info()
    
    def _get_material_value(self):
        material_value = 0
        for piece in self._board.piece_map().values():
            material_value += piece_values[piece.symbol()]
        
        return material_value

    def _get_opponent_move(self):
        # maximize opponent piece balance
        moves = list(self._board.legal_moves)
        max_move = None
        max_value = np.NINF

        before_material_value = self._get_material_value()
        for move in moves:
            self._board.push(move)
            if self._board.result() == '0-1':  # opponent win
                self._board.pop()
                return move
            after_material_value = self._get_material_value()
            value = 1 * (after_material_value - before_material_value)
            value += np.random.randn() / 1e+7  # noise
            if value >= max_value:
                max_value = value
                max_move = move
            self._board.pop()  # undo
        
        return max_move

    def step(self, action):
        before_material_value = self._get_material_value()

        move = self._action_id_dict[action]
        self._board.push(move)

        if self._board.result() != '*':
            observation = self._get_obs()
            reward = self.win_reward if self._board.result() == '1-0' else self.draw_reward  # draw reward : 0
            terminated = True
            truncated = False
            info = self._get_info()

            return observation, reward, terminated, truncated, info
        
        # opponent policy: greedy
        move = self._get_opponent_move()
        self._board.push(move)

        if self._board.result() != '*':
            observation = self._get_obs()
            reward = self.lose_reward if self._board.result() == '0-1' else self.draw_reward
            terminated = True
            truncated = False
            info = self._get_info()

            return observation, reward, terminated, truncated, info

        observation = self._get_obs()
        after_material_value = self._get_material_value()
        # heuristic: base negative reward
        auxiliary_reward = (after_material_value - before_material_value) * self._auxiliary_reward_factor
        reward = auxiliary_reward
        terminated = False
        truncated = False
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def render(self):
        return chess.svg.board(
            board=self._board,
            size=390,
            lastmove=self._board.peek() if self._board.move_stack else None,
            check=self._board.king(self._board.turn) if self._board.is_check() else None)
