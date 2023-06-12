import os

from .trainer.dqn import TrainDqn
from .trainer.dqnv2 import TrainDqnV2

import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf

import gym

import sys
import argparse

class DqnModel(keras.Model):
    def __init__(self, dqn_trainer):
        super().__init__(dqn_trainer)

        self.board_conv_1 = layers.Conv2D(7, kernel_size=(3, 3), padding='same', activation='relu')
        self.board_conv_2 = layers.Conv2D(5, kernel_size=(3, 3), padding='same', activation='relu')
        self.board_conv_3 = layers.Conv2D(3, kernel_size=(3, 3), activation='relu')

        self.action_dense_1 = layers.Dense(4096, activation='relu')
        self.action_dense_2 = layers.Dense(4096, activation='relu')
        self.action_dense_3 = layers.Dense(4096, activation='relu')

        self.full_dense_1 = layers.Dense(4096, activation='relu')
        self.full_dense_2 = layers.Dense(4096, activation='relu')

        self.output_dense = layers.Dense(dqn_trainer.num_actions, activation="softmax")

    def call(self, inputs):
        board_data = inputs[0]
        action_data = inputs[1]

        board_data = tf.reshape(board_data, (-1, 8, 8))
        board_data = self.board_conv_1(board_data)
        board_data = self.board_conv_2(board_data)
        board_data = self.board_conv_3(board_data)
        board_data = layers.Flatten()(board_data)

        action_data = self.action_dense_1(action_data)
        action_data = self.action_dense_2(action_data)
        action_data = self.action_dense_3(action_data)

        data = tf.concat([board_data, action_data], axis=1)
        data = self.full_dense_1(data)
        data = self.full_dense_2(data)

        return self.output_dense(data)

def create_q_model(dqn_trainer):
    board_input = layers.Input(shape=(8, 8))
    board_input = layers.Reshape((8, 8, 1))(board_input)

    cnn1 = layers.Conv2D(7, kernel_size=(3, 3), padding='same', activation='relu')(board_input)
    cnn1 = layers.Conv2D(5, kernel_size=(3, 3), padding='same', activation='relu')(cnn1)
    cnn1 = layers.Conv2D(3, kernel_size=(3, 3), activation='relu')(cnn1)

    # action_input = layers.Input(shape=(4096,))
    # action_dense = layers.Dense(4096, activation='relu')(action_input)
    # action_dense = layers.Dense(4096, activation='relu')(action_dense)
    # action_dense = layers.Dense(4096, activation='relu')(action_dense)

    flatten = layers.Flatten()(cnn1)
    # concat = layers.Concatenate()([flatten, action_dense])

    # dense = layers.Dense(4096, activation='relu')(concat)
    dense = layers.Dense(4096, activation='relu')(flatten)
    dense = layers.Dense(4096, activation='relu')(dense)

    action = layers.Dense(dqn_trainer.num_actions, activation="linear")(dense)
    # return keras.Model(inputs={'board_input': board_input, 'action_input': action_input}, outputs=action)
    return keras.Model(inputs={'board_input': board_input}, outputs=action)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', dest='mode', type=str, default='train', help='모드를 설정합니다.')
    parser.add_argument('--load_model', '-l', dest='model_path', type=str, default=None, help='로드할 모델의 경로를 설정합니다. 형식: 폴더명_에피소드 예) 4_4000')
    parser.add_argument('--batch_size', '-b', dest='batch_size', type=int, default=32, help='배치 사이즈를 설정합니다.')
    parser.add_argument('--epsilon', '-e', dest='epsilon', type=float, default=1.0, help='앱실론 값을 설정합니다.')
    parser.add_argument('--max_memory_length', '-M', dest='max_memory_length', type=int, default=10000, help='메모리에 최대로 저장할 개수를 설정합니다.')
    parser.add_argument('--epsilon_greedy_frames', '-g', dest='epsilon_greedy_frames', type=int, default=1000000, help='앱실론의 감소 속도를 조절합니다. 클수록 작아짐.')
    parser.add_argument('--epsilon_random_frames', '-r', dest='epsilon_random_frames', type=int, default=5000, help='랜덤으로 결정하는 프래임의 수를 조절합니다.')
    parser.add_argument('--alphabeta_depth', '-d', dest='alphabeta_depth', type=int, default=6, help='alpha-beta 가지치기 최대 깊이를 설정합니다.')
    parser.add_argument('--load_params', '-p', dest='load_params', type=int, default=0, help='모델에 저장되어있던 파라미터를 로드합니다.')
    parser.add_argument('--gpu', '-G', dest='gpu', type=int, default=1, help='gpu 사용 여부를 결정합니다.')
    parser.add_argument('--max_episodes', '-E', dest='max_episodes', type=int, default=10000, help='학습을 반복할 에피소드 개수를 설정합니다.')
    
    args = parser.parse_args()
    
    if not args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    dqn_trainer = TrainDqnV2(
        batch_size=args.batch_size,
        max_memory_length=args.max_memory_length,
        epsilon_greedy_frames=args.epsilon_greedy_frames,
        epsilon_random_frames=args.epsilon_random_frames,
        epsilon=args.epsilon,
        alphabeta_depth=args.alphabeta_depth,
        max_episodes=args.max_episodes
    )

    dqn_trainer.check_device(use_cpu_force=not args.gpu)
    dqn_trainer.set_env('ml_chess_env:ChessGreedyEnv')

    if args.model_path is None:
        dqn_trainer.set_models(create_q_model(dqn_trainer), create_q_model(dqn_trainer))
    else:
        dqn_trainer.load_model(args.model_path, args.load_params)
    
    if args.mode == 'train':
        dqn_trainer.train()
    else:
        dqn_trainer.evaluate()
