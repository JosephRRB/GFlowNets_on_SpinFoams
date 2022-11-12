import tensorflow as tf


class PolicyNetwork(tf.keras.Model):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.layer_1 = tf.keras.layers.Dense(self.n_hidden)
        self.layer_1.build(self.n_input)
        self.layer_2 = tf.keras.layers.Dense(self.n_hidden)
        self.layer_2.build(self.n_hidden)

        self.output_layer_1 = tf.keras.layers.Dense(self.n_output)
        self.output_layer_1.build(self.n_hidden)
        self.output_layer_2 = tf.keras.layers.Dense(self.n_output)
        self.output_layer_2.build(self.n_hidden)

    def call(self, input):
        hidden = self.layer_1(input)
        hidden = tf.keras.activations.tanh(hidden)
        hidden = self.layer_2(hidden)
        hidden = tf.keras.activations.tanh(hidden)

        