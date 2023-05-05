import os

import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.abspath(__file__ + "/../../")


class BaseSpinFoam:
    def __init__(self, n_boundary_intertwiners):
        self.n_boundary_intertwiners = n_boundary_intertwiners

    @tf.function
    def get_spinfoam_amplitudes(self, amplitudes, boundary_intertwiners):
        pass


class StarModelSpinFoam(BaseSpinFoam):
    def __init__(self):
        super().__init__(n_boundary_intertwiners=20)

    @staticmethod
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, ) * 5, dtype=tf.float64),
        tf.TensorSpec(shape=(None, 4), dtype=tf.int32),
    ])
    def _get_amplitude_per_star_edge(amplitudes, boundary_intertwiners_per_edge):
        amplitude = tf.gather_nd(amplitudes, boundary_intertwiners_per_edge)
        return amplitude

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, ) * 5, dtype=tf.float64),
        tf.TensorSpec(shape=(None, 20), dtype=tf.int32),
    ])
    def get_spinfoam_amplitudes(self, amplitudes, boundary_intertwiners):
        vertex_1 = self._get_amplitude_per_star_edge(
            amplitudes, boundary_intertwiners[:, :4]
        )
        vertex_2 = self._get_amplitude_per_star_edge(
            amplitudes, boundary_intertwiners[:, 4:8]
        )
        vertex_3 = self._get_amplitude_per_star_edge(
            amplitudes, boundary_intertwiners[:, 8:12]
        )
        vertex_4 = self._get_amplitude_per_star_edge(
            amplitudes, boundary_intertwiners[:, 12:16]
        )
        vertex_5 = self._get_amplitude_per_star_edge(
            amplitudes, boundary_intertwiners[:, 16:20]
        )

        star_amplitudes = tf.einsum(
            "abcde, ie, id, ic, ib, ia -> i",
            amplitudes, vertex_1, vertex_2, vertex_3, vertex_4, vertex_5
        )
        return star_amplitudes


class SingleVertexSpinFoam(BaseSpinFoam):
    def __init__(self):
        super().__init__(n_boundary_intertwiners=5)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, ) * 5, dtype=tf.float64),
        tf.TensorSpec(shape=(None, 5), dtype=tf.int32),
    ])
    def get_spinfoam_amplitudes(self, amplitudes, boundary_intertwiners):
        return tf.gather_nd(amplitudes, boundary_intertwiners)


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

        spinfoam_model: BaseSpinFoam
                    Python class representing a specific spinfoam model

                     n_boundary_intertwiners
                        - stores the number of boundary intertwiners
                        - represents the number of dimensions for the
                            environment grid

                    get_spinfoam_amplitudes()
                        - function that calculates the amplitudes of a specific
                            spinfoam model
                        - will be used to calculate the rewards for each position
                            in the environment grid
    """
    def __init__(self, spin_j, spinfoam_model: BaseSpinFoam):
        self.spinfoam_model = spinfoam_model
        self.grid_dimension = self.spinfoam_model.n_boundary_intertwiners
        self.spin_j = float(spin_j)
        self.grid_length = int(2 * self.spin_j + 1)
        self.single_vertex_amplitudes = tf.convert_to_tensor(
            _load_vertex_amplitudes(self.spin_j), dtype=tf.float64
        )
        scale = tf.math.sqrt(
            tf.math.reduce_sum(tf.math.square(self.single_vertex_amplitudes))
        )
        self.scaled_amplitudes = self.single_vertex_amplitudes / scale

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
    def reset_for_forward_sampling(self, batch_size):
        """Generate positions at the hypergrid origin of size batch_size"""
        positions = tf.zeros(
            shape=(batch_size, self.grid_dimension),
            dtype=tf.int32
        )
        return positions

    @staticmethod
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ])
    def step_forward(current_position, forward_action):
        new_position = current_position + forward_action[:, :-1]
        return new_position

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def get_rewards(self, positions):
        """Get the corresponding rewards for positions"""
        rewards = tf.reshape(
            self._get_squared_spinfoam_amplitudes(
                self.scaled_amplitudes, positions
            ),
            shape=(-1, 1)
        )
        return rewards

    @tf.function(input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ])
    def _get_squared_spinfoam_amplitudes(self, amplitudes, positions):
        return tf.math.square(
            self.spinfoam_model.get_spinfoam_amplitudes(amplitudes, positions)
        )

    @tf.function
    def reset_for_backward_sampling(self, batch_size):
        """Generate random positions in the hypergrid of size batch_size"""
        positions = tf.random.uniform(
            shape=(batch_size, self.grid_dimension),
            minval=0, maxval=self.grid_length, dtype=tf.int32
        )
        return positions

    @staticmethod
    @tf.function
    def step_backward(current_position, back_action):
        new_position = current_position - back_action
        return new_position


def _load_vertex_amplitudes(spin_j):
    vertex = np.load(f"{ROOT_DIR}/data/EPRL_vertices/Python/Dl_20/vertex_j_{spin_j}.npz")
    return vertex
