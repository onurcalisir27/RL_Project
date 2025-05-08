# Waypoint-Based RL with TensorFlow

Implementation of "Waypoint-Based Reinforcement Learning for Robot Manipulation Tasks" using TensorFlow.

## Setup
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training:
   ```bash
   python main.py --task Lift --train --method ours
   ```
4. Run evaluation:
   ```bash
   python main.py --task Lift --evaluate --method ours --model_path models/Lift/test
   ```

## Tasks
- Lift: Pick up a block.
- Extend to Stack, Door, etc., in `config.py` and `robosuite_env.py`.

## Logs
- TensorBoard: `tensorboard --logdir runs/`