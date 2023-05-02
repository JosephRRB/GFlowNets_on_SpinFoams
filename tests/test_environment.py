from core.environment import SpinFoamEnvironment, _load_vertex_amplitudes


def test_loaded_amplitudes_have_correct_size():
    spin_j = 3.5
    vertex_amplitudes = _load_vertex_amplitudes(spin_j)

    grid_length = int(2*spin_j + 1)
    assert vertex_amplitudes.shape == (grid_length, )*5


def test_loaded_amplitude_values_are_correct():
    spin_j = 3.0
    vertex_amplitudes = _load_vertex_amplitudes(spin_j)

    i1, i2, i3, i4, i5 = 0, 3, 0, 2, 0
    # From Python_notebook.ipynb
    expected_amplitude = -5.071973704515683e-13
    vertex_amplitude = vertex_amplitudes[i1, i2, i3, i4, i5]

    assert vertex_amplitude == expected_amplitude


def test_stored_squared_amplitudes_are_correct():
    spin_j = 3
    env = SpinFoamEnvironment(
        spin_j=spin_j
    )

    i1, i2, i3, i4, i5 = 0, 3, 0, 2, 0
    # From Python_notebook.ipynb
    expected_amplitude = -5.071973704515683e-13
    expected_squared_amplitude = expected_amplitude**2
    stored_squared_amplitude = env.squared_amplitudes[i1, i2, i3, i4, i5]

    assert stored_squared_amplitude == expected_squared_amplitude