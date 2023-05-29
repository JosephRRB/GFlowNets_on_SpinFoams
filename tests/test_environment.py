import tensorflow as tf
import numpy as np

from core.environment import SpinFoamEnvironment, SingleVertexSpinFoam, StarModelSpinFoam


def test_loaded_amplitudes_have_correct_size():
    spin_j = 3.5
    env = SpinFoamEnvironment(
        spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j)
    )
    vertex_amplitudes = env.spinfoam_model.single_vertex_amplitudes

    grid_length = int(2*spin_j + 1)
    assert vertex_amplitudes.shape == (grid_length, )*5


def test_single_vertex_amplitude_values_are_correct():
    spin_j = 3.0
    env = SpinFoamEnvironment(
        spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j)
    )

    i1, i2, i3, i4, i5 = 0, 3, 0, 2, 0
    positions = tf.constant([[i1, i2, i3, i4, i5]])
    vertex_amplitude = env.spinfoam_model.get_spinfoam_amplitudes(positions)

    # From Python_notebook.ipynb
    expected_amplitude = -5.071973704515683e-13

    assert vertex_amplitude == expected_amplitude


def test_single_vertex_rewards_are_correct():
    spin_j = 3
    env = SpinFoamEnvironment(
        spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j)
    )
    i1, i2, i3, i4, i5 = 0, 3, 0, 2, 0
    positions = tf.constant([[i1, i2, i3, i4, i5]])
    # From Python_notebook.ipynb
    expected_amplitude = -5.071973704515683e-13
    expected_amplitude = tf.constant(expected_amplitude, dtype=tf.float64)

    reward = env.get_rewards(positions)

    # Define rewards as square of amplitudes
    expected_reward = tf.math.square(expected_amplitude)

    assert reward == expected_reward


def test_log_of_star_amplitudes_squared_are_correct():
    spin_j = 3
    env = SpinFoamEnvironment(
        spinfoam_model=StarModelSpinFoam(spin_j=spin_j)
    )
    positions = tf.constant([
        [0, 1, 4, 3, 4, 0, 1, 4, 3, 2, 0, 1, 4, 3, 0, 0, 1, 4, 3, 0],
        [2, 0, 1, 4, 3, 0, 0, 1, 4, 3, 0, 0, 1, 4, 3, 4, 0, 1, 4, 3]
    ])

    vertex_amplitudes = env.spinfoam_model.single_vertex_amplitudes
    star_amplitudes = env.spinfoam_model.get_spinfoam_amplitudes(positions)
    log_sq_star_amplitudes = tf.math.log(tf.square(star_amplitudes))

    # From Python_notebook.ipynb
    def star_reward_optimized(tensor, indices, optimize_path=False):
        return np.square(
            np.einsum(
                'abcde, e, d, c, b, a ->', tensor,
                tensor[indices[0], indices[1], indices[2], indices[3], :],
                tensor[indices[4], indices[5], indices[6], indices[7], :],
                tensor[indices[8], indices[9], indices[10], indices[11], :],
                tensor[indices[12], indices[13], indices[14], indices[15], :],
                tensor[indices[16], indices[17], indices[18], indices[19], :],
                optimize=optimize_path
            )
        )

    expected_sq_star_amplitude_0 = star_reward_optimized(
        vertex_amplitudes.numpy(), positions[0, :].numpy()
    )
    expected_sq_star_amplitude_1 = star_reward_optimized(
        vertex_amplitudes.numpy(), positions[1, :].numpy()
    )

    expected_log_sq_star_ampl_0 = np.log(expected_sq_star_amplitude_0)
    expected_log_sq_star_ampl_1 = np.log(expected_sq_star_amplitude_1)

    np.testing.assert_almost_equal(
        log_sq_star_amplitudes[0], expected_log_sq_star_ampl_0
    )
    np.testing.assert_almost_equal(
        log_sq_star_amplitudes[1], expected_log_sq_star_ampl_1
    )


def test_star_amplitude_log_rewards_are_correct():
    spin_j = 3
    env = SpinFoamEnvironment(
        spinfoam_model=StarModelSpinFoam(spin_j=spin_j)
    )
    positions = tf.constant([
        [0, 1, 4, 3, 4, 0, 1, 4, 3, 2, 0, 1, 4, 3, 0, 0, 1, 4, 3, 0],
        [2, 0, 1, 4, 3, 0, 0, 1, 4, 3, 0, 0, 1, 4, 3, 4, 0, 1, 4, 3]
    ])
    rewards = env.get_rewards(positions)
    log_rewards = tf.math.log(rewards)

    # Define rewards as square of scaled amplitudes
    vertex_amplitudes = env.spinfoam_model.single_vertex_amplitudes

    # From Python_notebook.ipynb
    def star_reward_optimized(tensor, indices, optimize_path=False):
        return np.square(
            np.einsum(
                'abcde, e, d, c, b, a ->', tensor,
                tensor[indices[0], indices[1], indices[2], indices[3], :],
                tensor[indices[4], indices[5], indices[6], indices[7], :],
                tensor[indices[8], indices[9], indices[10], indices[11], :],
                tensor[indices[12], indices[13], indices[14], indices[15], :],
                tensor[indices[16], indices[17], indices[18], indices[19], :],
                optimize=optimize_path
            )
        )

    expected_log_rewards_0 = np.log(star_reward_optimized(
        vertex_amplitudes.numpy(), positions[0, :].numpy()
    ))
    expected_log_rewards_1 = np.log(star_reward_optimized(
        vertex_amplitudes.numpy(), positions[1, :].numpy()
    ))

    np.testing.assert_almost_equal(
        log_rewards[0, 0], expected_log_rewards_0
    )
    np.testing.assert_almost_equal(
        log_rewards[1, 0], expected_log_rewards_1
    )