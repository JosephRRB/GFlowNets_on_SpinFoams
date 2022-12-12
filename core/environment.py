import numpy as np
import tensorflow as tf


class HypergridEnvironment:
    def __init__(self, grid_dimension, grid_length, environment_mode="test_grid"):
        self.grid_dimension = grid_dimension
        self.grid_length = grid_length
        if environment_mode == "test_grid":
            self.rewards = _generate_grid_rewards(
                grid_dimension, grid_length, 0.001, 1, 1
            )
        elif environment_mode == "spinfoam_vertex":
            spin_j = (grid_length - 1) / 2
            vertex_amplitudes = tf.convert_to_tensor(
                _load_vertex_amplitudes(spin_j), dtype=tf.float64
            )
            self.squared_amplitudes = tf.math.square(vertex_amplitudes)
            scale = tf.math.reduce_sum(self.squared_amplitudes)
            self.rewards = tf.cast(self.squared_amplitudes/scale, dtype=tf.float32)
        else:
            NotImplementedError(
                "'environment_mode' must either be 'test_grid' or 'spinfoam_vertex'")

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


def _load_vertex_amplitudes(spin_j):
    vertex = np.load(f"../data/EPRL_vertices/python/vertex_j_{spin_j}.npz")
    return vertex


def _generate_grid_rewards(grid_dimension, grid_length, r0, r1, r2):
    # grid_length >= 7
    coords_transf = tf.math.abs(tf.range(grid_length, dtype=tf.float64) / (grid_length - 1) - 0.5)

    mid_level_1d = tf.cast(tf.math.greater(coords_transf, 0.25), dtype=tf.float64)
    high_level_1d = tf.cast(
        tf.math.logical_and(
            tf.math.greater(coords_transf, 0.3), tf.math.less(coords_transf, 0.4)),
        dtype=tf.float64
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