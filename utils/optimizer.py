import numpy as np
from scipy.optimize import minimize, LinearConstraint
import tensorflow as tf

def optimize_waypoint(reward_model, prev_traj, objs, config, curr_wp, load_model, models, n_inits=5):
    """Optimize waypoint using SLSQP."""
    state_dim = config['task']['env']['state_dim']
    action_space = config['task']['task']['action_space']
    
    def objective(wp):
        traj = np.concatenate([prev_traj, wp]) if prev_traj.size else wp
        input_data = np.concatenate([traj, objs])
        if load_model:
            reward = 0
            for model in models[curr_wp]:
                reward += model.predict(input_data[np.newaxis, :], verbose=0)[0][0]
            reward /= len(models[curr_wp])
        else:
            reward = reward_model.predict(input_data[np.newaxis, :], verbose=0)[0][0]
        return -reward

    lincon = LinearConstraint(np.eye(state_dim), -action_space, action_space)
    best_wp, best_val = None, float('inf')
    
    for _ in range(n_inits):
        init_wp = np.random.uniform(-action_space, action_space, state_dim)
        result = minimize(objective, init_wp, method='SLSQP', constraints=[lincon], options={'eps': 1e-6, 'maxiter': 1e6})
        if result.fun < best_val:
            best_val = result.fun
            best_wp = result.x
    return best_wp