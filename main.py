import argparse
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam
import pickle
import os
from tensorboardX import SummaryWriter
from config import CONFIG
from environment.robosuite_env import RobosuiteEnv
from models.reward_model import RewardModel
from models.ppo_model import PPOModel
from models.sac_model import SACModel
from utils.controller import get_action
from utils.optimizer import optimize_waypoint
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def push(self, traj, objs, reward):
        self.buffer.append((traj, objs, reward))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        trajs, objs, rewards = [], [], []
        for i in indices:
            t, o, r = self.buffer[i]
            trajs.append(t)
            objs.append(o)
            rewards.append(r)
        return np.array(trajs), np.array(objs), np.array(rewards)

    def __len__(self):
        return len(self.buffer)

def train(config, method='ours'):
    env = RobosuiteEnv(config)
    state_dim = config['task']['env']['state_dim']
    wp_steps = config['task']['env']['wp_steps']
    num_wp = config['num_wp']
    epoch_wp = config['task']['task']['epoch_wp']
    batch_size = config['task']['task']['batch_size']
    exploration_epoch = config['task']['task']['exploration_epoch']
    ensemble_sampling_epoch = config['task']['task']['ensemble_sampling_epoch']
    averaging_noise_epoch = config['task']['task']['averaging_noise_epoch']
    rand_reset_epoch = config['task']['task']['rand_reset_epoch']
    run_name = config['run_name']
    save_name = f"{config['task']['name']}/{run_name}"
    os.makedirs(f"models/{save_name}", exist_ok=True)
    writer = SummaryWriter(f"runs/{method}_{run_name}")

    if method == 'ours':
        models = [RewardModel(state_dim * i + 3) for i in range(1, num_wp + 1) for _ in range(config['algorithm']['num_models'])]
        # Build models
        for model in models:
            model.build((None, model.input_dim))
        learned_models = [[] for _ in range(num_wp)]
        optimizer = Adam(config['algorithm']['learning_rate'])
    elif method == 'ppo':
        model = PPOModel(state_dim, state_dim)
        model.build((None, state_dim))
        optimizer = Adam(1e-3)
    elif method == 'sac':
        model = SACModel(state_dim, state_dim)
        model.build((None, state_dim))
        optimizer = Adam(1e-3)

    memory = ReplayBuffer()
    save_data = {'episode': [], 'reward': []}
    total_steps = 0
    best_traj = np.random.uniform(-config['task']['task']['action_space'], config['task']['task']['action_space'], state_dim * num_wp)
    best_reward = -float('inf')

    for wp_id in range(1, num_wp + 1):
        obs, objs = env.reset(get_objs=True)
        curr_models = models[(wp_id-1) * config['algorithm']['num_models']:wp_id * config['algorithm']['num_models']]

        for i_episode in tqdm(range(epoch_wp)):
            if np.random.rand() < 0.05 and i_episode < rand_reset_epoch and i_episode > 1:
                idx = np.random.randint(config['algorithm']['num_models'])
                curr_models[idx] = RewardModel(state_dim * wp_id + 3)
                print(f"RESET MODEL {idx}")

            obs, objs = env.reset(get_objs=True)
            state = env.get_state(obs)
            traj = []

            for idx in range(wp_id):
                load_model = idx != wp_id - 1
                if load_model:
                    curr_wp_models = learned_models[idx]
                else:
                    curr_wp_models = curr_models
                    model_idx = np.random.randint(len(curr_models)) if i_episode < ensemble_sampling_epoch else None
                wp = optimize_waypoint(
                    curr_models[model_idx] if model_idx is not None else curr_models[0],
                    np.array(traj).flatten() if traj else np.array([]),
                    objs, config, idx, load_model, learned_models, config['n_inits']
                )
                if idx == wp_id - 1:
                    if i_episode <= exploration_epoch:
                        wp = np.random.uniform(-config['task']['task']['action_space'], config['task']['task']['action_space'], state_dim)
                    if np.random.rand() < 0.5 and i_episode > ensemble_sampling_epoch and i_episode < averaging_noise_epoch:
                        wp[-1] *= -1  # Noise to gripper
                        print("NOISE ADDED TO GRIPPER")
                    if np.random.rand() < 0.5 and i_episode > ensemble_sampling_epoch and i_episode < averaging_noise_epoch:
                        wp[:3] += np.random.normal(0, 0.05, 3)  # Noise to pose
                        print("NOISE ADDED TO POSE")
                traj.append(wp)

            traj_full = np.array(traj).flatten()
            traj_mat = np.reshape(traj_full, (wp_id, state_dim))[:, :state_dim-1] + state[:state_dim-1]
            gripper_mat = np.reshape(traj_full, (wp_id, state_dim))[:, state_dim-1:]

            if method != 'ours':
                traj_full, _ = model(tf.convert_to_tensor([state], dtype=tf.float32))
                traj_full = traj_full.numpy()[0]
                traj_mat = np.reshape(traj_full, (wp_id, state_dim))[:, :state_dim-1] + state[:state_dim-1]
                gripper_mat = np.reshape(traj_full, (wp_id, state_dim))[:, state_dim-1:]

            episode_reward = 0
            train_reward = 0
            time_s = 0

            for timestep in range(wp_id * wp_steps):
                if config['render']:
                    env.render()
                state = env.get_state(obs)
                wp_idx = timestep // wp_steps
                action = get_action(config['task']['name'], wp_idx, state, traj_mat, gripper_mat, time_s, timestep, config)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                if wp_idx == wp_id - 1:
                    train_reward += reward
                time_s += 1
                if time_s >= wp_steps:
                    time_s = 1
                total_steps += 1

            memory.push(traj_full, objs, episode_reward)
            save_data['episode'].append(i_episode)
            save_data['reward'].append(episode_reward)

            if method == 'ours' and len(memory) > batch_size:
                for _ in range(config['algorithm']['model_updates']):
                    trajs, objs, rewards = memory.sample(batch_size)
                    inputs = np.concatenate([trajs, objs], axis=-1)
                    for model in curr_models:
                        loss = model.train_step(inputs, rewards[:, np.newaxis], optimizer)
                    writer.add_scalar('critic_loss', loss, total_steps)
            elif method == 'ppo':
                trajs, objs, rewards = memory.sample(1)
                states = np.array([state])
                advantages = rewards - model.critic(states[np.newaxis, :]).numpy().flatten()
                loss = model.train_step(states[np.newaxis, :], trajs[np.newaxis, :], advantages, rewards, np.ones_like(rewards), clip_ratio=0.2)
                writer.add_scalar('ppo_loss', loss, total_steps)
            elif method == 'sac':
                trajs, objs, rewards = memory.sample(1)
                states = np.array([state])
                next_states = states  # Simplified
                dones = np.array([0])
                loss = model.train_step(states[np.newaxis, :], trajs[np.newaxis, :], rewards, next_states[np.newaxis, :], dones)
                writer.add_scalar('sac_loss', loss, total_steps)

            if train_reward > best_reward:
                best_reward = train_reward
                best_traj = traj_full
                if method == 'ours':
                    for i, model in enumerate(curr_models):
                        model.save_weights(f"models/{save_name}/model_{wp_id}_{i}.weights.h5")
                    learned_models[wp_id-1] = curr_models
                else:
                    model.save_weights(f"models/{save_name}/{method}.weights.h5")

            writer.add_scalar('reward', episode_reward, i_episode)
            print(f"wp_id: {wp_id}, Episode: {i_episode}, Reward: {round(episode_reward, 2)}, Predicted: {round(float(sum(model.predict(np.concatenate([traj_full, objs])[np.newaxis, :], verbose=0)[0] for model in curr_models)/len(curr_models)), 2) if method == 'ours' else 0}")
            pickle.dump(save_data, open(f"models/{save_name}/data.pkl", 'wb'))

def evaluate(config, method='ours'):
    env = RobosuiteEnv(config)
    state_dim = config['task']['env']['state_dim']
    wp_steps = config['task']['env']['wp_steps']
    num_wp = config['num_wp']
    num_eval = config['task']['task']['num_eval']
    save_name = f"{config['task']['name']}/{config['run_name']}"
    rewards = []

    if method == 'ours':
        models = [RewardModel(state_dim * num_wp + 3) for _ in range(config['algorithm']['num_models'])]
        for i, model in enumerate(models):
            model.load_weights(f"models/{save_name}/model_{num_wp}_{i}.weights.h5")
    else:
        model = PPOModel(state_dim, state_dim) if method == 'ppo' else SACModel(state_dim, state_dim)
        model.load_weights(f"models/{save_name}/{method}.weights.h5")

    for _ in range(num_eval):
        obs, objs = env.reset(get_objs=True)
        state = env.get_state(obs)
        episode_reward = 0

        if method == 'ours':
            traj = []
            for idx in range(num_wp):
                wp = optimize_waypoint(models[0], np.array(traj).flatten() if traj else np.array([]), objs, config, idx, True, [models], config['n_inits'])
                traj.append(wp)
            traj_full = np.array(traj).flatten()
            traj_mat = np.reshape(traj_full, (num_wp, state_dim))[:, :state_dim-1] + state[:state_dim-1]
            gripper_mat = np.reshape(traj_full, (num_wp, state_dim))[:, state_dim-1:]
        else:
            traj_full, _ = model(tf.convert_to_tensor([state], dtype=tf.float32))
            traj_full = traj_full.numpy()[0]
            traj_mat = np.reshape(traj_full, (num_wp, state_dim))[:, :state_dim-1] + state[:state_dim-1]
            gripper_mat = np.reshape(traj_full, (num_wp, state_dim))[:, state_dim-1:]

        time_s = 0
        for timestep in range(num_wp * wp_steps):
            state = env.get_state(obs)
            wp_idx = timestep // wp_steps
            action = get_action(config['task']['name'], wp_idx, state, traj_mat, gripper_mat, time_s, timestep, config)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            time_s += 1
            if time_s >= wp_steps:
                time_s = 1
        rewards.append(episode_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Evaluation Reward: {mean_reward} ± {std_reward}")
    pickle.dump({'episode': list(range(1, num_eval+1)), 'reward': rewards}, open(f"models/{save_name}/eval_data.pkl", 'wb'))
    return mean_reward, std_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Lift')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--method', type=str, default='ours', choices=['ours', 'ppo', 'sac'])
    args = parser.parse_args()

    CONFIG['task']['name'] = args.task
    if args.train:
        train(CONFIG, method=args.method)
    elif args.evaluate:
        evaluate(CONFIG, method=args.method)