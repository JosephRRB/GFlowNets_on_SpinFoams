import tensorflow as tf


class PolicyNetwork(tf.keras.Model):
    def __init__(
            self, main_layer_nodes, branch1_layer_nodes, branch2_layer_nodes
    ):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.main_layers = [
            tf.keras.layers.Dense(n_nodes) for n_nodes in main_layer_nodes
        ]
        self.branch1_layers = [
            tf.keras.layers.Dense(n_nodes) for n_nodes in branch1_layer_nodes
        ]
        self.branch2_layers = [
            tf.keras.layers.Dense(n_nodes) for n_nodes in branch2_layer_nodes
        ]
        # TODO: build layers
        # self.layer_1.build(self.n_input)


    def call(self, data_input):
        hidden = self.flatten(data_input)
        for main_layer in self.main_layers:
            hidden = main_layer(hidden)
            hidden = tf.keras.activations.tanh(hidden)

        b1_hidden = hidden
        for b1_layer in self.branch1_layers[:-1]:
            b1_hidden = b1_layer(b1_hidden)
            b1_hidden = tf.keras.activations.tanh(b1_hidden)

        b2_hidden = hidden
        for b2_layer in self.branch2_layers[:-1]:
            b2_hidden = b2_layer(b2_hidden)
            b2_hidden = tf.keras.activations.tanh(b2_hidden)

        output1 = self.branch1_layers[-1](b1_hidden)
        output2 = self.branch2_layers[-1](b2_hidden)
        return output1, output2
        