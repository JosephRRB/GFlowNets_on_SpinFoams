from core.environment import HypergridEnvironment, _load_vertex_amplitudes


def test_loaded_amplitudes_have_correct_size():
    spin_j = 3.5
    vertex_amplitudes = _load_vertex_amplitudes(spin_j)

    grid_length = int(2*spin_j + 1)
    assert vertex_amplitudes.shape == (grid_length, )*5


def test_loaded_amplitude_values_are_correct():
    spin_j = 3.0
    vertex_amplitudes = _load_vertex_amplitudes(spin_j)

    i1, i2, i3, i4, i5 = 0, 3, 0, 2, 0
    expected_amplitude = -7.021548787502716e-13
    vertex_amplitude = vertex_amplitudes[i1, i2, i3, i4, i5]

    assert vertex_amplitude == expected_amplitude


def test_stored_squared_amplitudes_are_correct():
    spin_j = 3
    env = HypergridEnvironment(
        grid_dimension=5, grid_length=2*spin_j+1, environment_mode="spinfoam_vertex"
    )

    i1, i2, i3, i4, i5 = 0, 3, 0, 2, 0
    expected_amplitude = -7.021548787502716e-13
    expected_squared_amplitude = expected_amplitude**2
    stored_squared_amplitude = env.squared_amplitudes[i1, i2, i3, i4, i5]

    assert stored_squared_amplitude == expected_squared_amplitude