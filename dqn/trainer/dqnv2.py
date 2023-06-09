import tensorflow as tf
import tensorflow.keras as keras

from ..tree.minimax import AlphBeta

import gym
import chess

import numpy as np

import time
from IPython.display import clear_output, display

import logging
import os
import shutil
import pickle

import gc

import tracemalloc
tracemalloc.start()

piece_mapper = {
    -1: 'r',
    -2: 'n',
    -3: 'b',
    -4: 'k',
    -5: 'q',
    -6: 'p',
    1: 'R',
    2: 'N',
    3: 'B',
    4: 'K',
    5: 'Q',
    6: 'P',
    0: '.'
}

class TrainDqnV2:
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
            max_memory_length=1000000,
            update_after_actions=4,
            update_target_network=10000,
            alphabeta_depth=4):

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
        
        self.tree_search = AlphBeta(depth=alphabeta_depth)

    def __init_logger(self):
        logs_path = './logs'
        shutil.rmtree(logs_path)
        os.makedirs(logs_path)

        self.__logger = logging.getLogger('dqn_trainer')
        self.__logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(u'%(asctime)s [%(levelname)s %(pathname)s] %(lineno)d: %(message)s')

        file_handler = logging.FileHandler('./logs/output.log')
        file_handler.setFormatter(formatter)

        formatter = logging.Formatter(u'[%(levelname)s] %(lineno)d: %(message)s')
        streaming_handler = logging.StreamHandler()
        streaming_handler.setFormatter(formatter)

        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(streaming_handler)

    def check_device(self, use_cpu_force=False):
        if use_cpu_force:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) and not use_cpu_force:
            self.__logger.info(f'Use GPU\tname: {gpus[0].name}')
            self.device = gpus[0].name.replace('physical_device:', '')
        else:
            cpu = tf.config.list_physical_devices('CPU')[0]
            self.__logger.info(f'Use CPU\tname: {cpu.name}')
            self.device = cpu.name.replace('physical_device:', '')

    def set_env(self, env):
        self.env = gym.make(env)
        self.__logger.info(f'Set Env: {env}')

    def set_models(self, model, model_target):
        model.summary()
        self.model = model
        self.model_target = model_target
        self.__logger.info(f'set models')
        
    def load_model(self, path, load_params=0):
        a, b = path.split('_')
        self.model = tf.keras.models.load_model(os.path.join('./models/', a, f'model.{b}'))
        self.model_target = tf.keras.models.load_model(os.path.join('./models/', a, f'model_target.{b}'))
        if load_params:
            with open(os.path.join('./models/', a, f'model.{b}', 'params.pkl'), 'rb') as f:
                self.gamma, self.epsilon, self.epsilon_min, self.epsilon_max, self.epsilon_interval, self.batch_size, \
                self.max_steps_per_episode, self.max_episodes, self.num_actions, self.learning_rate, self.epsilon_greedy_frames, \
                self.epsilon_random_frames, self.max_memory_length, self.update_after_actions, self.update_target_network, self.tree_search = \
                    pickle.load(f)

    def __init_train_variables(self):
        self.running_reward = 0
        self.episode_count = 0
        self.frame_count = 0
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.action_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []

        self.optimizer = keras.optimizers.experimental.RMSprop(learning_rate=self.learning_rate, clipnorm=1.0)
        self.loss_function = keras.losses.Huber()

    def __convert_state(self, board):
        return np.array(board).reshape((8, 8))

    def __get_greedy_epsilon(self, state, mask):
        if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
            action = np.random.choice([i for i in range(self.num_actions) if mask[i] == 1])
        else:
            # print("greedy")
            action_probs = self.model({'board_input': np.expand_dims(state, 0), 'action_input': np.expand_dims(mask, 0)}, training=False)
            
            valid_probs = [(i, action_probs[0][i]) for i in range(self.num_actions) if mask[i] == 1]
            valid_probs = sorted(valid_probs, key=lambda x: -x[1])
            actions = [i[0] for i in valid_probs[:5]]
            action = self.tree_search.get_alphabeta_action(actions, self.env)

            # valid_probs = [(i, action_probs[0][i]) for i in range(self.num_actions) if mask[i] == 1]
            # idx, val = max(valid_probs, key=lambda e: e[1])
            # action = np.random.choice([i for i, prob in valid_probs if prob >= val])

        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return action

    def __get_greedy_action(self, state, mask):
        # print("greedy action")
        action_probs = self.model({'board_input': np.expand_dims(state, 0), 'action_input': np.expand_dims(mask, 0)}, training=False)

        # valid_probs = [(i, action_probs[0][i]) for i in range(self.num_actions) if mask[i] == 1]
        # idx, val = max(valid_probs, key=lambda e: e[1])
        # action = np.random.choice([i for i, prob in valid_probs if prob >= val])
        
        valid_probs = [(i, action_probs[0][i]) for i in range(self.num_actions) if mask[i] == 1]
        valid_probs = sorted(valid_probs, key=lambda x: -x[1])
        actions = [i[0] for i in valid_probs[:5]]
        action = self.tree_search.get_alphabeta_action(actions, self.env)

        return action

    def __make_fen(self, line):
        result = ''
        empty_count = 0
        for i in line:
            # i = i[0]
            if i == 0:
                empty_count += 1
                continue
            if empty_count != 0:
                result += str(empty_count)
                empty_count = 0
            result += piece_mapper.get(i)
        if empty_count != 0:
            result += str(empty_count)
        return result

    def __sample_batch(self, batch_size):
        indices = np.random.choice(range(len(self.done_history)), size=batch_size)

        state_sample = np.array([self.state_history[i] for i in indices])
        state_next_sample = np.array([self.state_next_history[i] for i in indices])
        action_next_sample = np.array([self.action_next_history[i] for i in indices])
        rewards_sample = [self.rewards_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]
        done_sample = tf.convert_to_tensor(
            [float(self.done_history[i]) for i in indices]
        )

        return state_sample, state_next_sample, rewards_sample, action_sample, done_sample, action_next_sample

    def train(self):
        snapshot = tracemalloc.take_snapshot()
        
        self.__init_train_variables()
        model_save_path = './models/'
        folder_list = []
        for d in os.listdir(model_save_path):
            try:
                folder_list.append(int(d))
            except:
                continue
        folder_list = sorted(folder_list)
        if folder_list:
            model_save_path = os.path.join(model_save_path, str(folder_list[-1] + 1))
        else:
            model_save_path = os.path.join(model_save_path, '0')
        os.makedirs(model_save_path)

        start = time.time()
        for _ in range(self.max_episodes):
            board, info = self.env.reset()
            state = self.__convert_state(board)
            action_mask = info['action_mask']
            episode_reward = 0
            
            for timestep in range(1, self.max_steps_per_episode):
                self.frame_count += 1

                action = self.__get_greedy_epsilon(state, action_mask)

                board, reward, done, _, info = self.env.step(action)
                state_next = self.__convert_state(board)
                action_mask = info['action_mask']

                episode_reward += reward

                self.action_history.append(action)
                self.state_history.append(state)
                self.state_next_history.append(state_next)

                board = chess.Board()
                board.set_board_fen('/'.join(map(self.__make_fen, state_next.tolist())))
                legal_moves = list(board.legal_moves)
                action_next_mask = np.zeros(shape=(4096,), dtype=np.bool8)
                for move in legal_moves:
                    if move.promotion is not None and move.promotion != 5:
                        continue

                    action_id = move.from_square * 64 + move.to_square
                    action_next_mask[action_id] = True
                
                self.action_next_history.append(action_next_mask)
                self.done_history.append(done)
                self.rewards_history.append(reward)
                state = state_next

                if self.frame_count % self.update_after_actions == 0 and len(self.done_history) > self.batch_size:
                    state_sample, state_next_sample, rewards_sample, action_sample, done_sample, action_next_sample = \
                        self.__sample_batch(self.batch_size)

                    # print('predict')
                    # print(state_next_sample.dtype)
                    # print(state_next_sample.shape)
                    # print(type(state_next_sample))
                    future_rewards = self.model_target.predict({'board_input': state_next_sample, 'action_input': action_next_sample}, verbose=0)

                    updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)

                    masks = tf.one_hot(action_sample, self.num_actions)

                    with tf.GradientTape() as tape:
                        # print('gradient')
                        q_values = self.model({'board_input': state_sample, 'action_input': masks})
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        loss = self.loss_function(updated_q_values, q_action)

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    gc.collect()

                if self.frame_count % self.update_target_network == 0:
                    self.model_target.set_weights(self.model.get_weights())
                    self.__logger.info(f"model sync - running reward: {self.running_reward} at episode {self.episode_count}, frame count {self.frame_count}")

                if len(self.rewards_history) > self.max_memory_length:
                    del self.rewards_history[:1]
                    del self.state_history[:1]
                    del self.state_next_history[:1]
                    del self.action_next_history[:1]
                    del self.action_history[:1]
                    del self.done_history[:1]

                if done:
                    break

            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                del self.episode_reward_history[:1]
            self.running_reward = np.mean(self.episode_reward_history)

            self.episode_count += 1

            if self.episode_count % 10 == 0:
                self.__logger.info(f"# episode = {self.episode_count}:\tavg. reward = {self.running_reward}\tepsilon:{self.epsilon}" +
                                   f"\ttime = {time.time() - start}sec\tmemory usage: {len(self.rewards_history) / self.max_memory_length * 100}%")
                start = time.time()
                
            if self.episode_count % 500 == 0:
                lines = []
                top_stats = tracemalloc.take_snapshot().compare_to(snapshot, 'lineno')
                for stat in top_stats[:10]:
                    lines.append(str(stat))
                self.__logger.debug(f"top 10 memory increse:\n\t" + '\n\t'.join(lines))
                snapshot = tracemalloc.take_snapshot()

            if self.episode_count % 1000 == 0:
                self.model.save(os.path.join(model_save_path, 'model.{}'.format(self.episode_count)))
                self.model_target.save(os.path.join(model_save_path, 'model_target.{}'.format(self.episode_count)))
                with open(os.path.join(model_save_path, 'model.{}'.format(self.episode_count), 'params.pkl'), 'wb') as f:
                    pickle.dump([self.gamma, self.epsilon, self.epsilon_min, self.epsilon_max, self.epsilon_interval, self.batch_size,
                                    self.max_steps_per_episode, self.max_episodes, self.num_actions, self.learning_rate, self.epsilon_greedy_frames,
                                    self.epsilon_random_frames, self.max_memory_length, self.update_after_actions, self.update_target_network, self.tree_search], f)
                # tf.keras.backend.clear_session()
                # self.load_model(f'{model_save_path.split("/")[-1]}_{self.episode_count}')



        self.model.save(os.path.join(model_save_path, 'model.final'))
        self.model_target.save(os.path.join(model_save_path, 'model_target.final'))
        with open(os.path.join(model_save_path, 'model.final', 'done_history.pkl'), 'wb') as f:
            pickle.dump([self.gamma, self.epsilon, self.epsilon_min, self.epsilon_max, self.epsilon_interval, self.batch_size,
                            self.max_steps_per_episode, self.max_episodes, self.num_actions, self.learning_rate, self.epsilon_greedy_frames,
                            self.epsilon_random_frames, self.max_memory_length, self.update_after_actions, self.update_target_network, self.tree_search], f)
        self.evaluate()

    def evaluate(self):
        win = 0
        loss = 0
        draw = 0
        trial = 100
        for timestep in range(trial):
            board, info = self.env.reset()
            done = False
            count = 0
            while 1:
                count += 1
                state = self.__convert_state(board)
                action_mask = info['action_mask']

                move = self.__get_greedy_action(state, action_mask)
                board, reward, done, _, info = self.env.step(move)

                # clear_output()
                # display(self.env.render())
                # self.env.print_board()
                # time.sleep(1)

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
