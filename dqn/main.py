import os

from trainer.dqn import TrainDqn

import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf

import gym

def create_q_model(dqn_trainer):
    inputs = layers.Input(shape=(64,))
    layer1 = layers.Dense(128, activation="relu")(inputs)
    layer2 = layers.Dense(128, activation="relu")(layer1)
    action = layers.Dense(dqn_trainer.num_actions, activation="linear")(layer2)
    return keras.Model(inputs=inputs, outputs=action)

if __name__ == '__main__':
    dqn_trainer = TrainDqn(batch_size=256)

    dqn_trainer.check_device(use_cpu_force=False)

    dqn_trainer.set_env('ml_chess_env:ChessGreedyEnv')

    dqn_trainer.set_models(create_q_model(dqn_trainer), create_q_model(dqn_trainer))
    dqn_trainer.train()
    dqn_trainer.evaluate()
