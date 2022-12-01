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


class HypergridEnvironment:
    def __init__(self, grid_dimension, grid_length):
        self.grid_dimension = grid_dimension
        self.grid_length = grid_length

    @tf.function
    def reset_for_backward_sampling(self, batch_size):
        positions = tf.random.uniform(
            shape=(batch_size, self.grid_dimension),
            minval=0, maxval=self.grid_length, dtype=tf.int32
        )
        return positions

    @tf.function
    def reset_for_forward_sampling(self, batch_size):
        positions = tf.zeros(
            shape=(batch_size, self.grid_dimension),
            dtype=tf.int32
        )
        return positions

    @staticmethod
    @tf.function
    def step_backward(current_position, back_action):
        new_position = current_position - back_action
        return new_position

    @staticmethod
    @tf.function
    def step_forward(current_position, forward_action):
        new_position = current_position + forward_action[:, :-1]
        return new_position
