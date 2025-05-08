import tensorflow as tf
from tensorflow.keras import layers, models

class SACModel(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(SACModel, self).__init__()
        self.actor = models.Sequential([
            layers.Dense(64, input_dim=state_dim, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_dim, activation='tanh')
        ])
        self.critic1 = models.Sequential([
            layers.Dense(64, input_dim=state_dim + action_dim, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        self.critic2 = models.Sequential([
            layers.Dense(64, input_dim=state_dim + action_dim, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, states):
        action = self.actor(states)
        return action

    def train_step(self, states, actions, rewards, next_states, dones, alpha=0.2):
        with tf.GradientTape() as tape:
            next_actions = self.actor(next_states)
            next_inputs = tf.concat([next_states, next_actions], axis=-1)
            next_q1 = self.critic1(next_inputs)
            next_q2 = self.critic2(next_inputs)
            next_q = tf.minimum(next_q1, next_q2)
            target_q = rewards + (1 - dones) * 0.99 * next_q
            inputs = tf.concat([states, actions], axis=-1)
            q1 = self.critic1(inputs)
            q2 = self.critic2(inputs)
            critic_loss = tf.reduce_mean(tf.square(target_q - q1) + tf.square(target_q - q2))
            actor_loss = tf.reduce_mean(alpha * tf.math.log(self.actor(states)) - self.critic1(inputs))
            loss = critic_loss + actor_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(1e-3)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss