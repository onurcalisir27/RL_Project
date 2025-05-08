import tensorflow as tf
from tensorflow.keras import layers, models

class PPOModel(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(PPOModel, self).__init__()
        self.actor = models.Sequential([
            layers.Dense(64, input_dim=state_dim, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_dim, activation='tanh')
        ])
        self.critic = models.Sequential([
            layers.Dense(64, input_dim=state_dim, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, states):
        action = self.actor(states)
        value = self.critic(states)
        return action, value

    def train_step(self, states, actions, advantages, returns, old_probs, clip_ratio=0.2):
        with tf.GradientTape() as tape:
            new_actions, values = self(states)
            new_probs = tf.reduce_sum(tf.square(new_actions - actions), axis=-1)
            ratio = new_probs / (old_probs + 1e-10)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            loss = actor_loss + 0.5 * critic_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(1e-3)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss