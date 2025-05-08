import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
import numpy as np
from scipy.spatial.transform import Rotation as R

class RobosuiteEnv:
    def __init__(self, config):
        self.config = config
        self.env_name = config['task']['name']
        controller_config = load_composite_controller_config(controller="BASIC", robot=config['task']['env']['robot'])
        self.env = suite.make(
            env_name=self.env_name,
            robots=config['task']['env']['robot'],
            controller_configs=controller_config,
            has_renderer=config['render'],
            reward_shaping=True,
            control_freq=10,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            initialization_noise=None
        )

    def reset(self, get_objs=False):
        obs = self.env.reset()
        if get_objs:
            if self.env_name == 'Lift':
                objs = obs['cube_pos']
            elif self.env_name == 'Stack':
                objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1)
            elif self.env_name == 'NutAssembly':
                objs = obs['RoundNut_pos']
            elif self.env_name == 'PickPlace':
                objs = obs[self.config['object'] + '_pos']
            elif self.env_name == 'Door':
                objs = np.concatenate((obs['door_pos'], obs['handle_pos']), axis=-1)
            else:
                objs = np.zeros(3)  # Placeholder
            return obs, objs
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def get_state(self, obs):
        if self.env_name == 'Door':
            pos = obs['robot0_eef_pos']
            ang = R.from_quat(obs['robot0_eef_quat']).as_euler('xyz', degrees=False)
            return np.concatenate([pos, ang])
        pos = obs['robot0_eef_pos']
        gripper = np.array([0.])  # Simplified
        return np.concatenate([pos, gripper])

    def render(self):
        self.env.render()