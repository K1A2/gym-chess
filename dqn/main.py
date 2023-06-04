from trainer.dqn import TrainDqn

import tensorflow.keras.layers as layers

import gym

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(64, ))

    # "Dense" is the basic form of a neural network layer
    # "Dense" stands for fully connected layer, which means each neuron in a layer
    # receives input from all neurons of the previous layer.
    layer1 = layers.Dense(128, activation="relu")(inputs)
    layer2 = layers.Dense(128, activation="relu")(layer1)
    action = layers.Dense(num_actions, activation="linear")(layer2)

    return keras.Model(inputs=inputs, outputs=action)

if __name__ == '__main__':
    dqn_trainer = TrainDqn()
    device = dqn_trainer.check_device()

    dqn_trainer.set_env('ml_chess_env:ChessGreedyEnv')
