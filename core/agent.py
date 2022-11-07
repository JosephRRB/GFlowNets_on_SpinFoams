import tensorflow as tf

class Agent:
    def __init__(self, spin_j):
        self.spin_j = spin_j
        self.intertwiner_dim = int(2*spin_j + 1)

    def action(self, state):
        pass


def _encode_state(state, intertwiner_dim):
    encoded = tf.reshape(
        tf.one_hot(state, intertwiner_dim, dtype=tf.float64),
        shape=(1, -1)
    )
    return encoded