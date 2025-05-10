import numpy as np

def get_waypoint_action(wp_idx, state, traj_mat, gripper_mat, time_s, timestep, wp_steps, gripper_steps):
    error = traj_mat[wp_idx, :] - state
    if timestep < 10:
        full_action = np.array(list(10. * error) + [0.] * (6 - len(state)) + [-1.])
    elif time_s >= wp_steps - gripper_steps:
        full_action = np.array([0.] * 6 + list(gripper_mat[wp_idx]))
    else:
        full_action = np.array(list(10. * error) + [0.] * (6 - len(state)) + [0.])
    return full_action