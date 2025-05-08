CONFIG = {
    'task': {
        'name': 'Lift',
        'env': {
            'robot': 'Panda',
            'state_dim': 4,  # 3D position + 1D gripper
            'gripper_steps': 10,
            'wp_steps': 50,
            'use_latch': False
        },
        'task': {
            'action_space': 0.5,  # Waypoint bounds
            'batch_size': 30,
            'epoch_wp': 200,
            'exploration_epoch': 75,
            'ensemble_sampling_epoch': 100,
            'averaging_noise_epoch': 175,
            'rand_reset_epoch': 150,
            'num_eval': 100
        }
    },
    'num_wp': 2,
    'render': False,
    'n_inits': 5,
    'run_name': 'test',
    'object': '',
    'algorithm': {
        'num_models': 10,
        'learning_rate': 0.001,
        'hidden_size': 128,
        'model_updates': 100
    }
}