import numpy as np

def get_action(env_name, wp_idx, state, traj_mat, gripper_mat, time_s, timestep, config):
    """Generate action to track waypoint using impedance control."""
    wp_steps = config['task']['env']['wp_steps']
    gripper_steps = config['task']['env']['gripper_steps']
    state_dim = config['task']['env']['state_dim']

    error = traj_mat[wp_idx, :] - state[:state_dim-1]  # Position error
    if timestep < 10:
        action = np.concatenate([10. * error, [0.] * (6 - (state_dim-1)), [-1.]])  # Open gripper
    elif time_s >= wp_steps - gripper_steps:
        action = np.concatenate([[0.] * 6, gripper_mat[wp_idx]])  # Gripper action
    else:
        action = np.concatenate([10. * error, [0.] * (6 - (state_dim-1)), [0.]])  # Track waypoint
    return action