import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from scipy.spatial.transform import Rotation as R
import gymnasium as gym

class RobosuiteEnv:
    def __init__(self, env_name, robot, object_type, render, use_latch=False):
        controller_config = load_controller_config(default_controller="OSC_POSE")
        self.env = suite.make(
            env_name=env_name,
            robots=robot,
            controller_configs=controller_config,
            has_renderer=render,
            reward_shaping=True,
            control_freq=10,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            initialization_noise=None,
            single_object_mode=2,
            object_type=object_type,
            use_latch=use_latch,
        )
        self.env_name = env_name
        self.object_type = object_type

    def reset(self, get_objs=False):
        obs = self.env.reset()
        if get_objs:
            if self.env_name == 'Lift':
                objs = obs['cube_pos']
            elif self.env_name == 'Stack':
                objs = np.concatenate((obs['cubeA_pos'], obs['cubeB_pos']), axis=-1)
            elif self.env_name == 'NutAssembly':
                nut = 'RoundNut'
                objs = obs[nut + '_pos']
            elif self.env_name == 'PickPlace':
                objs = obs[self.object_type + '_pos']
            elif self.env_name == 'Door':
                objs = np.concatenate((obs['door_pos'], obs['handle_pos']), axis=-1)
            return obs, objs
        return obs

    def get_state(self, obs, objs=None):
        if self.env_name == 'Door':
            robot_pos = obs['robot0_eef_pos']
            robot_ang = R.from_quat(obs['robot0_eef_quat']).as_euler('xyz', degrees=False)
            state = np.concatenate((robot_pos, robot_ang), axis=-1)
        else:
            state = obs['robot0_eef_pos']
        if objs is not None:
            state = np.concatenate((state, objs))
        return state

    def get_action_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()