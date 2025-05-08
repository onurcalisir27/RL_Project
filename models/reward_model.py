import tensorflow as tf
from tensorflow.keras import layers, models

class RewardModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(RewardModel, self).__init__()
        self.input_dim = input_dim
        self.model = models.Sequential([
            layers.Dense(128, input_dim=input_dim),
            layers.LeakyReLU(alpha=0.01),
            layers.Dense(128),
            layers.LeakyReLU(alpha=0.01),
            layers.Dense(1)
        ])

    def call(self, inputs, training=False):
        return self.model(inputs)

    def train_step(self, x, y, optimizer):
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = tf.reduce_mean(tf.square(predictions - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss