import os
from .tree.minimax import AlphBeta
import argparse
import gym

num_actions = 64 * 64

def get_action(search, mask, env):
    valid_probs = [(i, 1) for i in range(num_actions) if mask[i] == 1]
    valid_probs = sorted(valid_probs, key=lambda x: -x[1])
    actions = [i[0] for i in valid_probs]
    action = search.get_alphabeta_action(actions, env)
    return action

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphabeta_depth', '-d', dest='alphabeta_depth', type=int, default=6, help='alpha-beta 가지치기 최대 깊이를 설정합니다.')
    
    args = parser.parse_args()
    
    search = AlphBeta(depth=args.alphabeta_depth)
    
    env = gym.make('ml_chess_env:ChessGreedyEnv')
    
    win = 0
    loss = 0
    draw = 0
    trial = 100
    for timestep in range(trial):
        board, info = env.reset()
        done = False
        count = 0
        while 1:
            count += 1
            action_mask = info['action_mask']

            move = get_action(search, action_mask, env)
            board, reward, done, _, info = env.step(move)

            if done:
                if info['result'] == '1-0':
                    win += 1
                    print(f'game {timestep}\tresult: win\tcount{count}')
                elif info['result'] == '0-1':
                    loss += 1
                    print(f'game {timestep}\tresult: loss\tcount{count}')
                else:
                    draw += 1
                    print(f'game {timestep}\tresult: draw\tcount{count}')
                break
    print(f'trial: {trial}\twin: {win}\tloss: {loss}\tdraw: {draw}')
    print(f'win rate: {win / trial * 100 }%')
