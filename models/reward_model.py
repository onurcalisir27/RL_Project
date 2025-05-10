import tensorflow as tf
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import os
import random
from tqdm import tqdm
import pickle
from tensorboardX import SummaryWriter
import datetime

class RNetwork(tf.keras.Model):
    def __init__(self, traj_dim, hidden_dim):
        super(RNetwork, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, input_shape=(traj_dim,), activation='leaky_relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_dim, activation='leaky_relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, traj):
        return self.model(traj)

class WeightClipper(tf.keras.callbacks.Callback):
    def on_batch_end(self):
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.kernel
                weights.assign(tf.clip_by_value(weights, 0, float('inf')))

class MyMemory:
    def __init__(self):
        self.buffer = []
        self.position = 0

    def push(self, traj, reward):
        self.buffer.append(None)
        self.buffer[self.position] = (traj, reward)
        self.position += 1

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        trajs, rewards = map(np.stack, zip(*batch))
        return trajs, rewards

    def __len__(self):
        return len(self.buffer)

class WaypointMethod:
    def __init__(self, state_dim, objs, wp_id, save_name, config):
        self.state_dim = state_dim
        self.objs = np.array(objs, dtype=np.float32)
        self.wp_id = wp_id
        self.best_wp = []
        self.n_inits = 5
        self.lr = 1e-3
        self.hidden_size = 128
        self.n_models = 10
        self.models = []
        self.learned_models = []
        self.n_eval = 100

        self.action_dim = config['task']['task']['action_space']
        self.exploration_epoch = config['task']['task']['exploration_epoch']
        self.ensemble_sampling_epoch = config['task']['task']['ensemble_sampling_epoch']
        self.averaging_noise_epoch = config['task']['task']['averaging_noise_epoch']

        # Initialize and build models
        for _ in range(self.n_models):
            critic = RNetwork(self.state_dim * self.wp_id + len(objs), hidden_dim=self.hidden_size)
            critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
            # Build model with dummy input
            dummy_traj = tf.zeros((1, self.state_dim * self.wp_id + len(objs)))
            critic(dummy_traj)
            self.models.append(critic)

        save_dir = 'models/' + save_name
        for wp_id in range(1, self.wp_id):
            models = []
            for idx in range(self.n_models):
                critic = RNetwork(wp_id * self.state_dim + len(objs), hidden_dim=self.hidden_size)
                critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
                # Build model with dummy input
                dummy_traj = tf.zeros((1, wp_id * self.state_dim + len(objs)))
                critic(dummy_traj)
                critic.load_weights(os.path.join(save_dir, f'model_{wp_id}_{idx}.weights.h5'))
                models.append(critic)
            self.learned_models.append(models)

        self.best_traj = self.action_dim * (np.random.rand(self.state_dim * self.wp_id) - 0.5)
        self.best_reward = -np.inf
        self.lincon = LinearConstraint(np.eye(self.state_dim), -self.action_dim, self.action_dim)

    def traj_opt(self, i_episode, objs):
        self.reward_idx = random.choice(range(self.n_models))
        self.objs = np.array(objs, dtype=np.float32)
        self.curr_episode = i_episode
        self.traj = []
        for idx in range(1, self.wp_id + 1):
            min_cost = np.inf
            self.load_model = idx != self.wp_id
            self.curr_wp = idx - 1
            if idx == self.wp_id and i_episode <= self.exploration_epoch:
                self.best_wp = self.action_dim * (np.random.rand(self.state_dim) - 0.5)
            else:
                for t_idx in range(self.n_inits):
                    xi0 = (np.copy(self.best_traj[self.curr_wp * self.state_dim:self.curr_wp * self.state_dim + self.state_dim]) +
                           np.random.normal(0, 0.1, size=self.state_dim) if t_idx != 0 else
                           np.copy(self.best_traj[self.curr_wp * self.state_dim:self.curr_wp * self.state_dim + self.state_dim]))
                    res = minimize(self.get_cost, xi0, method='SLSQP', constraints=self.lincon,
                                 options={'eps': 1e-6, 'maxiter': 1e6})
                    if res.fun < min_cost:
                        min_cost = res.fun
                        self.best_wp = res.x
                if idx == self.wp_id and np.random.rand() < 0.5 and i_episode > self.ensemble_sampling_epoch and i_episode < self.averaging_noise_epoch:
                    tqdm.write("NOISE ADDED TO GRIPPER")
                    self.best_wp[-1] *= -1
                if idx == self.wp_id and np.random.rand() < 0.5 and i_episode > self.ensemble_sampling_epoch and i_episode < self.averaging_noise_epoch:
                    tqdm.write("NOISE ADDED TO POSE")
                    self.best_wp[:3] += np.random.normal(0, 0.05, 3)
            self.traj.append(self.best_wp)
        return np.array(self.traj).flatten()

    def set_init(self, traj, reward):
        self.best_reward = reward
        self.best_traj = np.copy(traj)

    def get_cost(self, traj):
        traj_learnt = np.array(self.traj).flatten()
        traj_combined = np.concatenate((traj_learnt, traj))
        traj_combined = np.concatenate((traj_combined, self.objs))
        traj_tensor = tf.convert_to_tensor(traj_combined, dtype=tf.float32)[tf.newaxis, :]
        reward = self.get_reward(traj_tensor)
        return -reward

    def get_reward(self, traj):
        if self.load_model:
            models = self.learned_models[self.curr_wp]
            loss = 0
            for critic in models:
                loss += critic(traj).numpy()[0, 0]
            return loss / self.n_models
        else:
            if self.curr_episode < self.ensemble_sampling_epoch:
                critic = self.models[self.reward_idx]
                return critic(traj).numpy()[0, 0]
            else:
                loss = 0
                for critic in self.models:
                    loss += critic(traj).numpy()[0, 0]
                return loss / self.n_models

    def get_avg_reward(self, traj):
        traj = np.concatenate((traj, self.objs))
        traj_tensor = tf.convert_to_tensor(traj, dtype=tf.float32)[tf.newaxis, :]
        reward = 0
        for critic in self.models:
            reward += critic(traj_tensor).numpy()[0, 0]
        return reward / self.n_models

    def update_parameters(self, memory, batch_size):
        loss = np.zeros(self.n_models)
        for idx, critic in enumerate(self.models):
            loss[idx] = self.update_critic(critic, memory, batch_size)
        return np.mean(loss)

    def update_critic(self, critic, memory, batch_size):
        trajs, rewards = memory.sample(batch_size)
        trajs = tf.convert_to_tensor(trajs, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)[:, tf.newaxis]
        with tf.GradientTape() as tape:
            rhat = critic(trajs)
            loss = tf.reduce_mean(tf.square(rhat - rewards))
        grads = tape.gradient(loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(grads, critic.trainable_variables))
        return loss.numpy()

    def reset_model(self, idx):
        critic = RNetwork(self.wp_id * self.state_dim + len(self.objs), hidden_dim=self.hidden_size)
        critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        # Build model with dummy input
        dummy_traj = tf.zeros((1, self.wp_id * self.state_dim + len(self.objs)))
        critic(dummy_traj)
        self.models[idx] = critic
        tqdm.write(f"RESET MODEL {idx}")

    def save_model(self, save_name: str):
        save_dir = 'models/' + save_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx, critic in enumerate(self.models):
            # Ensure model is built before saving
            if not critic.built:
                dummy_traj = tf.zeros((1, self.wp_id * self.state_dim + len(self.objs)))
                critic(dummy_traj)
            critic.save_weights(os.path.join(save_dir, f'model_{self.wp_id}_{idx}.weights.h5'))

    def load_model(self, save_name: str, wp_id: int, idx: int):
        save_dir = 'models/' + save_name
        return tf.keras.models.load_model(os.path.join(save_dir, f'model_{wp_id}_{idx}.weights.h5'))

class TrainWaypoint:
    def __init__(self, config):
        self.config = config
        self.env_name = config['task']['name']
        self.num_wp = config['num_wp']
        self.render = config['render']
        self.n_inits = config['n_inits']
        self.run_name = config['run_name']
        self.object = None if config['object'] == '' else config['object']
        self.robot = config['task']['env']['robot']
        self.state_dim = config['task']['env']['state_dim']
        self.gripper_steps = config['task']['env']['gripper_steps']
        self.wp_steps = config['task']['env']['wp_steps']
        self.use_latch = config['task']['env']['use_latch']
        self.batch_size = config['task']['task']['batch_size']
        self.epoch_wp = config['task']['task']['epoch_wp']
        self.rand_reset_epoch = config['task']['task']['rand_reset_epoch']
        self.train()

    def get_action(self, wp_idx, state, traj_mat, gripper_mat, time_s, timestep):
        error = traj_mat[wp_idx, :] - state
        if timestep < 10:
            full_action = np.array(list(10. * error) + [0.] * (6 - len(state)) + [-1.])
        elif time_s >= self.wp_steps - self.gripper_steps:
            full_action = np.array([0.] * 6 + list(gripper_mat[wp_idx]))
        else:
            full_action = np.array(list(10. * error) + [0.] * (6 - len(state)) + [0.])
        return full_action

    def train(self):
        from environment.robosuite_env import RobosuiteEnv
        save_data = {'episode': [], 'reward': []}
        save_name = f"models/{self.env_name}/{self.run_name}" if self.object is None else f"models/{self.env_name}/{self.object}/{self.run_name}"
        env = RobosuiteEnv(self.env_name, self.robot, self.object, self.render, self.use_latch)
        wp_id = 1
        obs, objs = env.reset(get_objs=True)
        agent = WaypointMethod(state_dim=self.state_dim, objs=objs, wp_id=wp_id, save_name=save_name, config=self.config)
        memory = MyMemory()
        writer = SummaryWriter(f'runs/waypoint_{self.run_name}_{datetime.datetime.now().strftime("%H-%M")}')
        EPOCHS = self.epoch_wp * self.num_wp
        total_steps = 0
        for i_episode in tqdm(range(1, EPOCHS)):
            if i_episode % self.epoch_wp == 0:
                agent.save_model(save_name)
                wp_id += 1
                agent = WaypointMethod(state_dim=self.state_dim, objs=objs, wp_id=wp_id, save_name=save_name, config=self.config)
                memory = MyMemory()
            i_episode = i_episode % self.epoch_wp
            if np.random.rand() < 0.05 and i_episode < self.rand_reset_epoch and i_episode > 1:
                agent.reset_model(np.random.randint(10))
            episode_reward = 0
            obs, objs = env.reset(get_objs=True)
            traj_full = agent.traj_opt(i_episode, objs)
            state = env.get_state(obs)
            traj_mat = np.reshape(traj_full, (wp_id, self.state_dim))[:, :self.state_dim - 1] + state
            gripper_mat = np.reshape(traj_full, (wp_id, self.state_dim))[:, self.state_dim - 1:]
            if len(memory) > self.batch_size:
                for _ in range(100):
                    critic_loss = agent.update_parameters(memory, self.batch_size)
                    writer.add_scalar('model/critic_loss', critic_loss, total_steps)
            time_s = 0
            train_reward = 0
            for timestep in range(wp_id * self.wp_steps):
                if self.render:
                    env.render()
                state = env.get_state(obs)
                wp_idx = timestep // 50
                action = self.get_action(wp_idx, state, traj_mat, gripper_mat, time_s, timestep)
                time_s += 1
                if time_s >= 50:
                    time_s = 1
                obs, reward, _, _ = env.step(action)
                episode_reward += reward
                if timestep // 50 == wp_id - 1:
                    train_reward += reward
                total_steps += 1
            memory.push(np.concatenate((traj_full, objs)), episode_reward)
            save_data['episode'].append(i_episode)
            save_data['reward'].append(episode_reward)
            if train_reward > agent.best_reward:
                agent.set_init(traj_full, train_reward)
                agent.save_model(save_name)
            writer.add_scalar('reward', episode_reward, i_episode)
            tqdm.write(f"wp_id: {wp_id}, Episode: {i_episode}, Reward_full: {round(episode_reward, 2)}; Reward: {round(train_reward, 2)}, Predicted: {round(agent.get_avg_reward(traj_full), 2)}")
            pickle.dump(save_data, open(f'{save_name}/data.pkl', 'wb'))
        exit()

class EvaluateWaypoint:
    def __init__(self, config):
        self.config = config
        self.env_name = config['task']['name']
        self.num_wp = config['num_wp']
        self.render = config['render']
        self.n_inits = config['n_inits']
        self.run_name = config['run_name']
        self.object = None if config['object'] == '' else config['object']
        self.robot = config['task']['env']['robot']
        self.state_dim = config['task']['env']['state_dim']
        self.gripper_steps = config['task']['env']['gripper_steps']
        self.wp_steps = config['task']['env']['wp_steps']
        self.use_latch = config['task']['env']['use_latch']
        self.n_eval = config['task']['task']['num_eval']
        self.eval()

    def get_action(self, wp_idx, state, traj_mat, gripper_mat, time_s, timestep):
        error = traj_mat[wp_idx, :] - state
        if timestep < 10:
            full_action = np.array(list(10. * error) + [0.] * (6 - len(state)) + [-1.])
        elif time_s >= self.wp_steps - self.gripper_steps:
            full_action = np.array([0.] * 6 + list(gripper_mat[wp_idx]))
        else:
            full_action = np.array(list(10. * error) + [0.] * (6 - len(state)) + [0.])
        return full_action

    def eval(self):
        from environment.robosuite_env import RobosuiteEnv
        save_data = {'episode': [], 'reward': []}
        save_name = f"models/{self.env_name}/{self.run_name}" if self.object is None else f"models/{self.env_name}/{self.object}/{self.run_name}"
        env = RobosuiteEnv(self.env_name, self.robot, self.object, self.render, self.use_latch)
        wp_id = self.num_wp
        obs, objs = env.reset(get_objs=True)
        agent = WaypointMethod(state_dim=self.state_dim, objs=objs, wp_id=wp_id, save_name=save_name, config=self.config)
        for i_episode in tqdm(range(1, self.n_eval)):
            episode_reward = 0
            obs, objs = env.reset(get_objs=True)
            traj_full = agent.traj_opt(i_episode, objs)
            state = env.get_state(obs)
            traj_mat = np.reshape(traj_full, (wp_id, self.state_dim))[:, :self.state_dim - 1] + state
            gripper_mat = np.reshape(traj_full, (wp_id, self.state_dim))[:, self.state_dim - 1:]
            time_s = 0
            train_reward = 0
            for timestep in range(wp_id * self.wp_steps):
                if self.render:
                    env.render()
                state = env.get_state(obs)
                wp_idx = timestep // 50
                action = self.get_action(wp_idx, state, traj_mat, gripper_mat, time_s, timestep)
                time_s += 1
                if time_s >= 50:
                    time_s = 1
                obs, reward, _, _ = env.step(action)
                episode_reward += reward
                if timestep // 50 == wp_id - 1:
                    train_reward += reward
            save_data['episode'].append(i_episode)
            save_data['reward'].append(episode_reward)
            pickle.dump(save_data, open(f'{save_name}/eval_data.pkl', 'wb'))
        exit()