from core.agent import _encode_state
import tensorflow as tf


def test_state_is_correctly_encoded():
    state = tf.constant([
        [6, 3, 5, 0, 3]
    ], dtype=tf.int32)
    spin_j = 3.5
    intertwiner_dim = int(2*spin_j + 1)
    encoded_state = _encode_state(state, intertwiner_dim)

    encoded_i1 = [0, 0, 0, 0, 0, 0, 1, 0]
    encoded_i2 = [0, 0, 0, 1, 0, 0, 0, 0]
    encoded_i3 = [0, 0, 0, 0, 0, 1, 0, 0]
    encoded_i4 = [1, 0, 0, 0, 0, 0, 0, 0]
    encoded_i5 = [0, 0, 0, 1, 0, 0, 0, 0]
    expected_encoded_state = tf.constant([
        encoded_i1 + encoded_i2 + encoded_i3 + encoded_i4 + encoded_i5
    ], dtype=tf.float64)

    tf.debugging.assert_equal(encoded_state, expected_encoded_state)


def test_encoded_state_has_correct_dimensionality():
    state = tf.constant([
        [6, 3, 5, 0, 3]
    ], dtype=tf.int32)
    spin_j = 3
    intertwiner_dim = int(2 * spin_j + 1)
    encoded_state = _encode_state(state, intertwiner_dim)

    expected_dim = intertwiner_dim * 5

    assert encoded_state.shape == (1, expected_dim)