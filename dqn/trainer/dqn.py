import tensorflow as tf

import gym

import logging
import os
import shutil

class TrainDqn:
    def __init__(
            self,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_max=1.0,
            batch_size=32,
            max_steps_per_episode=10000,
            max_episodes=10000,
            num_actions=4096,
            learning_rate=0.001,
            epsilon_random_frames=50000,
            epsilon_greedy_frames=1000000,
            max_memory_length=100000,
            update_after_actions=4,
            update_target_network=10000):

        self.__init_logger()

        # Discount factor for past rewards
        self.gamma = gamma
        # Epsilon greedy parameter
        self.epsilon = epsilon
        # Minimum epsilon greedy parameter
        self.epsilon_min = epsilon_min
        # Maximum epsilon greedy parameter
        self.epsilon_max = epsilon_max
        # Rate at which to reduce chance of random action being taken
        self.epsilon_interval = self.epsilon_max - self.epsilon_min
        # Size of batch taken from replay buffer
        self.batch_size = batch_size
        self.max_steps_per_episode = max_steps_per_episode
        self.max_episodes = max_episodes

        self.num_actions = num_actions
        self.learning_rate = learning_rate

        # Number of frames to take random action and observe output
        self.epsilon_random_frames = epsilon_random_frames
        # Number of frames for exploration
        self.epsilon_greedy_frames = epsilon_greedy_frames
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        self.max_memory_length = max_memory_length
        # Train the model after 4 actions
        self.update_after_actions = update_after_actions
        # How often to update the target network
        self.update_target_network = update_target_network

    def __init_logger(self):
        logs_path = '../logs'
        shutil.rmtree(logs_path)
        os.makedirs(logs_path)

        self.__logger = logging.getLogger('dqn_trainer')
        self.__logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(u'%(asctime)s [%(levelname)s %(pathname)s] %(lineno)d: %(message)s')

        file_handler = logging.FileHandler('../logs/output.log')
        file_handler.setFormatter(formatter)

        formatter = logging.Formatter(u'[%(levelname)s] %(lineno)d: %(message)s')
        streaming_handler = logging.StreamHandler()
        streaming_handler.setFormatter(formatter)

        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(streaming_handler)

    def check_device(self, use_cpu_force=False):
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) and not use_cpu_force:
            self.__logger.info(f'Use GPU\tname: {gpus[0].name}')
            return gpus[0].name
        else:
            cpu = tf.config.list_physical_devices('CPU')[0]
            self.__logger.info(f'Use CPU\tname: {cpu[0].name}')
            return cpu.name

    def set_env(self, env):
        self.env = gym.make(env)
        self.__logger.info(f'Set Env: {env}')

    def set_models(self, model, target_model):
        model.summary()
        self.model = model
        self.target_model = target_model
        self.__logger.info(f'set models')
