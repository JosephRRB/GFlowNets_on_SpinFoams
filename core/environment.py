import os

import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.abspath(__file__ + "/../../")


class SpinFoamEnvironment:
    """
    This class creates the hypergrid environment which the GFlowNet agent
    interacts with. Loads the precalculated amplitudes for a single vertex,
    calculates the corresponding probabilities, and stores them in self.rewards

    Parameters:
    ----------
        spin_j:    (float)
                    Length of the grid in one dimension is 2*spin_j + 1. All
                    the lengths are equal in all dimensions

    TODO: Should allow user to use their own queryable reward function (without
            needing to precalculate all the rewards)
    """
    def __init__(self, spin_j):
        self.grid_dimension = 5
        self.spin_j = float(spin_j)
        self.grid_length = int(2 * self.spin_j + 1)
        vertex_amplitudes = tf.convert_to_tensor(
            _load_vertex_amplitudes(self.spin_j), dtype=tf.float64
        )
        self.squared_amplitudes = tf.math.square(vertex_amplitudes)
        norm = tf.math.reduce_sum(self.squared_amplitudes)
        normed_sq_ampl = self.squared_amplitudes / norm
        self.rewards = tf.cast(normed_sq_ampl, dtype=tf.float32)

    @tf.function
    def reset_for_forward_sampling(self, batch_size):
        """Generate positions at the hypergrid origin of size batch_size"""
        positions = tf.zeros(
            shape=(batch_size, self.grid_dimension),
            dtype=tf.int32
        )
        return positions

    @staticmethod
    @tf.function
    def step_forward(current_position, forward_action):
        new_position = current_position + forward_action[:, :-1]
        return new_position

    @tf.function
    def get_rewards(self, positions):
        """Get the corresponding rewards for positions"""
        rewards = tf.reshape(tf.gather_nd(self.rewards, positions), shape=(-1, 1))
        return rewards


def _load_vertex_amplitudes(spin_j):
    vertex = np.load(f"{ROOT_DIR}/data/EPRL_vertices/Python/Dl_20/vertex_j_{spin_j}.npz")
    return vertex
