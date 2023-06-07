import os

from trainer.dqn import TrainDqn
from trainer.dqnv2 import TrainDqnV2

import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf

import gym

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

        self.output_dense = layers.Dense(dqn_trainer.num_actions, activation="linear")

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

    action_input = layers.Input(shape=(4096,))
    action_dense = layers.Dense(4096, activation='relu')(action_input)
    action_dense = layers.Dense(4096, activation='relu')(action_dense)
    action_dense = layers.Dense(4096, activation='relu')(action_dense)

    flatten = layers.Flatten()(cnn1)
    concat = layers.Concatenate()([flatten, action_dense])

    dense = layers.Dense(4096, activation='relu')(concat)
    dense = layers.Dense(4096, activation='relu')(dense)

    action = layers.Dense(dqn_trainer.num_actions, activation="linear")(dense)
    return keras.Model(inputs={'board_input': board_input, 'action_input': action_input}, outputs=action)
    # return DqnModel(dqn_trainer)

if __name__ == '__main__':
    dqn_trainer = TrainDqnV2(
        batch_size=256,
        max_memory_length=500000,
        epsilon_greedy_frames=2000000
    )

    dqn_trainer.check_device(use_cpu_force=False)

    dqn_trainer.set_env('ml_chess_env:ChessGreedyEnv')

    dqn_trainer.set_models(create_q_model(dqn_trainer), create_q_model(dqn_trainer))
    dqn_trainer.train()
    dqn_trainer.evaluate()
