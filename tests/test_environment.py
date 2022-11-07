from core.environment import Environment, _load_vertex_amplitudes


def test_loaded_amplitudes_have_correct_size():
    spin_j = 3.5
    vertex_amplitudes = _load_vertex_amplitudes(spin_j)

    intertwiner_dim = int(2*spin_j + 1)
    assert vertex_amplitudes.shape == (intertwiner_dim, )*5


def test_loaded_amplitude_values_are_correct():
    spin_j = 3
    vertex_amplitudes = _load_vertex_amplitudes(spin_j)

    i1, i2, i3, i4, i5 = 0, 2, 0, 3, 0
    expected_amplitude = -5.11394413748117e-13
    vertex_amplitude = vertex_amplitudes[i1, i2, i3, i4, i5]

    assert vertex_amplitude == expected_amplitude


def test_stored_squared_amplitudes_are_correct():
    spin_j = 3
    env = Environment(spin_j)

    i1, i2, i3, i4, i5 = 0, 2, 0, 3, 0
    expected_amplitude = -5.11394413748117e-13
    expected_squared_amplitude = expected_amplitude**2
    stored_squared_amplitude = env.squared_vertex_amplitudes[i1, i2, i3, i4, i5]

    assert stored_squared_amplitude == expected_squared_amplitude