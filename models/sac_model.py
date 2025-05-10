import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import pickle
from tensorboardX import SummaryWriter
import datetime
import random

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

class QNetwork(tf.keras.Model):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.q1 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, input_shape=(num_inputs + num_actions,),
                                activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(1)
        ])
        self.q2 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, input_shape=(num_inputs + num_actions,),
                                activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, state, action):
        xu = tf.concat([state, action], axis=1)
        q1 = self.q1(xu)
        q2 = self.q2(xu)
        return q1, q2

class GaussianPolicy(tf.keras.Model):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.base = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, input_shape=(num_inputs,),
                                activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros'),
            tf.keras.layers.Dense(hidden_dim, activation='relu',
                                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer='zeros')
        ])
        self.mean_linear = tf.keras.layers.Dense(num_actions,
                                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                              bias_initializer='zeros')
        self.log_std_linear = tf.keras.layers.Dense(num_actions,
                                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                                 bias_initializer='zeros')
        if action_space is None:
            self.action_scale = tf.constant(1.0, dtype=tf.float32)
            self.action_bias = tf.constant(0.0, dtype=tf.float32)
        else:
            self.action_scale = tf.constant((action_space.high - action_space.low) / 2.0, dtype=tf.float32)
            self.action_bias = tf.constant((action_space.high + action_space.low) / 2.0, dtype=tf.float32)

    def call(self, state):
        x = self.base(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.call(state)
        std = tf.exp(log_std)
        normal = tf.random.normal(tf.shape(mean))
        x_t = mean + std * normal
        y_t = tf.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = -0.5 * ((x_t - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        log_prob -= tf.math.log(self.action_scale * (1 - y_t ** 2) + EPSILON)
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        mean = tf.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

def soft_update(target, source, tau):
    for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
        target_var.assign((1.0 - tau) * target_var + tau * source_var)

def hard_update(target, source):
    for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
        target_var.assign(source_var)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class SAC:
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.policy_type = args['policy']
        self.target_update_interval = args['target_update_interval']
        self.lr = args['lr']
        
        # Initialize and build critic
        self.critic = QNetwork(num_inputs, action_space.shape[0], args['hidden_size'])
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        # Build critic with dummy input
        dummy_state = tf.zeros((1, num_inputs))
        dummy_action = tf.zeros((1, action_space.shape[0]))
        self.critic(dummy_state, dummy_action)
        
        # Initialize and build critic target
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args['hidden_size'])
        self.critic_target.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        # Build critic target with dummy input
        self.critic_target(dummy_state, dummy_action)
        hard_update(self.critic_target, self.critic)
        
        # Initialize and build policy
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args['hidden_size'], action_space)
        self.policy.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        # Build policy with dummy input
        self.policy(dummy_state)

    def select_action(self, state, evaluate=False):
        state = tf.convert_to_tensor(state, dtype=tf.float32)[tf.newaxis, :]
        action, _, _ = self.policy.sample(state)
        return action.numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)[:, tf.newaxis]
        mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)[:, tf.newaxis]
        with tf.GradientTape(persistent=True) as tape:
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = tf.minimum(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
            qf1, qf2 = self.critic(state_batch, action_batch)
            qf1_loss = tf.reduce_mean(tf.square(qf1 - next_q_value))
            qf2_loss = tf.reduce_mean(tf.square(qf2 - next_q_value))
            qf_loss = qf1_loss + qf2_loss
            pi, log_pi, _ = self.policy.sample(state_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = tf.minimum(qf1_pi, qf2_pi)
            policy_loss = tf.reduce_mean(self.alpha * log_pi - min_qf_pi)
        critic_grads = tape.gradient(qf_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        return qf1_loss.numpy(), qf2_loss.numpy(), policy_loss.numpy()

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = f"checkpoints/sac_checkpoint_{env_name}_{suffix}"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        self.policy.save_weights(f"{ckpt_path}_policy.weights.h5")
        self.critic.save_weights(f"{ckpt_path}_critic.weights.h5")
        self.critic_target.save_weights(f"{ckpt_path}_critic_target.weights.h5")

    def load_checkpoint(self, ckpt_path, evaluate=False):
        self.policy.load_weights(f"{ckpt_path}_policy.weights.h5")
        self.critic.load_weights(f"{ckpt_path}_critic.weights.h5")
        self.critic_target.load_weights(f"{ckpt_path}_critic_target.weights.h5")

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.policy.save_weights(os.path.join(save_dir, "actor.weights.h5"))
        self.critic.save_weights(os.path.join(save_dir, "critic.weights.h5"))

class TrainSAC:
    def __init__(self, config):
        self.config = config
        self.env_name = config['task']['name']
        self.run_name = config['run_name']
        self.object = None if config['object'] == '' else config['object']
        self.render = config['render']
        self.num_steps = config['sac']['num_steps']
        self.start_steps = config['sac']['start_steps']
        self.num_episodes = config['sac']['num_episodes']
        self.args = config['sac']
        self.robot = config['task']['env']['robot']
        self.train()

    def train(self):
        from environment.robosuite_env import RobosuiteEnv
        save_data = {'episode': [], 'reward': []}
        save_name = f"models/{self.env_name}/{self.run_name}" if self.object is None else f"models/{self.env_name}/{self.object}/{self.run_name}"
        os.makedirs(save_name, exist_ok=True)
        env = RobosuiteEnv(self.env_name, self.robot, self.object, self.render)
        obs, objs = env.reset(get_objs=True)
        env_action_space = env.get_action_space()
        agent = SAC(len(obs['robot0_eef_pos']) + len(objs), env_action_space, self.args)
        memory = ReplayMemory(self.args['replay_size'])
        writer = SummaryWriter(f'runs/sac_{self.run_name}_{datetime.datetime.now().strftime("%H-%M")}')
        total_numsteps = 0
        updates = 0
        for i_episode in tqdm(range(1, self.num_episodes + 1)):
            episode_reward = 0
            episode_steps = 0
            obs, objs = env.reset(get_objs=True)
            state = env.get_state(obs, objs)
            for timestep in range(1, self.num_steps + 1):
                if self.render:
                    env.render()
                if total_numsteps < self.start_steps:
                    action = env_action_space.sample()
                else:
                    action = agent.select_action(state)
                if len(memory) > self.args['batch_size']:
                    for _ in range(self.args['updates_per_step']):
                        critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, self.args['batch_size'], updates)
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        updates += 1
                full_action = list(action[0:3]) + [0.] * 3 + [action[3]]
                full_action = np.array(full_action)
                obs, reward, _, _ = env.step(full_action)
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward
                next_objs = env.reset(get_objs=True)[1]
                next_state = env.get_state(obs, next_objs)
                memory.push(state, action, reward, next_state, 1.0)
                state = next_state
            save_data['episode'].append(i_episode)
            save_data['reward'].append(episode_reward)
            writer.add_scalar('reward', episode_reward, i_episode)
            tqdm.write(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {round(episode_reward, 2)}")
            pickle.dump(save_data, open(f"{save_name}/data.pkl", 'wb'))
            agent.save_checkpoint(self.env_name, ckpt_path=f"{save_name}/models")
        exit()

class EvaluateSAC:
    def __init__(self, config):
        self.config = config
        self.env_name = config['task']['name']
        self.run_name = config['run_name']
        self.object = None if config['object'] == '' else config['object']
        self.render = config['render']
        self.num_steps = config['sac']['num_steps']
        self.num_eval = config['sac']['num_eval']
        self.args = config['sac']
        self.robot = config['task']['env']['robot']
        self.eval()

    def eval(self):
        from environment.robosuite_env import RobosuiteEnv
        save_data = {'episode': [], 'reward': []}
        save_name = f"models/{self.env_name}/{self.run_name}" if self.object is None else f"models/{self.env_name}/{self.object}/{self.run_name}"
        os.makedirs(save_name, exist_ok=True)
        env = RobosuiteEnv(self.env_name, self.robot, self.object, self.render)
        obs, objs = env.reset(get_objs=True)
        env_action_space = env.get_action_space()
        agent = SAC(len(obs['robot0_eef_pos']) + len(objs), env_action_space, self.args)
        agent.load_checkpoint(f"{save_name}/models", evaluate=True)
        total_numsteps = 0
        for i_episode in tqdm(range(1, self.num_eval + 1)):
            episode_reward = 0
            episode_steps = 0
            obs, objs = env.reset(get_objs=True)
            state = env.get_state(obs, objs)
            for timestep in range(1, self.num_steps + 1):
                if self.render:
                    env.render()
                action = agent.select_action(state)
                full_action = list(action[0:3]) + [0.] * 3 + [action[3]]
                full_action = np.array(full_action)
                obs, reward, _, _ = env.step(full_action)
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward
                next_objs = env.reset(get_objs=True)[1]
                state = env.get_state(obs, next_objs)
            save_data['episode'].append(i_episode)
            save_data['reward'].append(episode_reward)
            pickle.dump(save_data, open(f"{save_name}/eval_reward.pkl", 'wb'))
            tqdm.write(f"Saved test data to {save_name}/eval_reward.pkl")
        exit()