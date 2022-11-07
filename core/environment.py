import numpy as np
import tensorflow as tf


class Environment:
    def __init__(self, spin_j):
        self.spin_j = spin_j
        vertex_amplitudes = tf.convert_to_tensor(
            _load_vertex_amplitudes(self.spin_j),
            dtype=tf.float64
        )
        self.squared_vertex_amplitudes = tf.math.square(vertex_amplitudes)
        self.current_state = None

    def reset(self):
        self.current_state = tf.zeros(shape=(1, 5), dtype=tf.int32)
        return self.current_state

    def step(self, action):
        pass


def _load_vertex_amplitudes(spin_j):
    vertex = np.load(f"../data/EPRL_vertices/vertex_j={float(spin_j)}.npz")
    return vertex

