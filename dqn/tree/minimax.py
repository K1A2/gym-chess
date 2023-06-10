import chess
import numpy as np
import sys

class AlphBeta:
    def __init__(self,
                 depth=4):
        self.depth = depth
        self.board = chess.Board()
        self.piece_values = {
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
        
    def __alphabeta_step(self, move):
        self.board.push(move)
        terminated = False
        if self.board.result() != '*':
            terminated = True
        return terminated, list(self.board.legal_moves)
    
    def __get_material_value(self):
        material_value = 0
        for piece in self.board.piece_map().values():
            material_value += self.piece_values[piece.symbol()]
        return material_value
    
    def get_alphabeta_action(self, action_maske, env):
        target_move = None
        def __alphabeta(depth, alpha, beta, maxPlayer, termination, moves):
            nonlocal target_move
            if depth == 0 or termination:
                return self.__get_material_value() * (-1 if maxPlayer else 1)
            if maxPlayer:
                for move in moves:
                    terminated, legal_moves = self.__alphabeta_step(move)
                    value = __alphabeta(depth - 1, alpha, beta, 0, terminated, legal_moves)
                    if alpha < value:
                        if depth == self.depth:
                            target_move = move
                        alpha = value
                    self.board.pop()
                    if beta <= alpha:
                        # print('alpha puring')
                        break
                return alpha
            else:
                for move in moves:
                    terminated, legal_moves = self.__alphabeta_step(move)
                    beta = __alphabeta(depth - 1, alpha, beta, 1, terminated, legal_moves)
                    self.board.pop()
                    if beta <= alpha:
                        # print('beta puring')
                        break
                return beta
        moves, move_to_action_idx = env.get_alphabeta_move(action_maske)
        self.board = chess.Board()
        self.board.set_fen(env.get_fen())
        __alphabeta(self.depth, -1e10, 1e10, 1, False, moves)
        return move_to_action_idx[target_move.uci()]