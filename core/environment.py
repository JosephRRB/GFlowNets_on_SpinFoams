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
        self.rewards = _generate_grid_rewards(
            grid_dimension, grid_length, 0.001, 1, 1
        )

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

    @tf.function
    def get_rewards(self, positions):
        rewards = tf.reshape(tf.gather_nd(self.rewards, positions), shape=(-1, 1))
        return rewards


def _generate_grid_rewards(grid_dimension, grid_length, r0, r1, r2):
    # grid_length >= 7
    coords_transf = tf.math.abs(tf.range(grid_length, dtype=tf.float32) / (grid_length - 1) - 0.5)

    mid_level_1d = tf.cast(tf.math.greater(coords_transf, 0.25), dtype=tf.float32)
    high_level_1d = tf.cast(
        tf.math.logical_and(
            tf.math.greater(coords_transf, 0.3), tf.math.less(coords_transf, 0.4)),
        dtype=tf.float32
    )

    mid_level = tf.reshape(mid_level_1d, shape=[-1] + [1] * (grid_dimension - 1))
    high_level = tf.reshape(high_level_1d, shape=[-1] + [1] * (grid_dimension - 1))

    for i in range(1, grid_dimension):
        shape = [1] * grid_dimension
        shape[i] = -1
        mid_level *= tf.reshape(mid_level_1d, shape=shape)
        high_level *= tf.reshape(high_level_1d, shape=shape)

    rewards = r0 + r1 * mid_level + r2 * high_level
    return rewards