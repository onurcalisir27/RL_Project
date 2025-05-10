import tensorflow as tf
import numpy as np
from scipy.optimize import minimize, LinearConstraint

def waypoint_optimize(cost_fn, xi0, state_dim, action_dim):
    lincon = LinearConstraint(np.eye(state_dim), -action_dim, action_dim)
    res = minimize(cost_fn, xi0, method='SLSQP', constraints=lincon, options={'eps': 1e-6, 'maxiter': 1e6})
    return res

def soft_update(target, source, tau):
    for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
        target_var.assign((1.0 - tau) * target_var + tau * source_var)

def hard_update(target, source):
    for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
        target_var.assign(source_var)