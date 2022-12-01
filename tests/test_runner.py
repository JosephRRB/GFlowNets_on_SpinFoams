import tensorflow as tf

from core.runner import Runner


def test_actions_correctly_correspond_to_backward_trajectories():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(grid_dimension=grid_dim, grid_length=grid_length)
    trajectories, backward_actions, forward_actions = runner.generate_backward_trajectories(batch_size)

    next_backward_positions = trajectories[:-1] - backward_actions[:-1]
    next_forward_positions = trajectories[1:] + forward_actions[1:, :, :-1]

    tf.debugging.assert_equal(trajectories[1:], next_backward_positions)
    tf.debugging.assert_equal(trajectories[:-1], next_forward_positions)


def test_backward_trajectories_do_not_go_out_of_bounds():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(grid_dimension=grid_dim, grid_length=grid_length)
    trajectories, _, _ = runner.generate_backward_trajectories(batch_size)

    tf.debugging.assert_greater_equal(trajectories, 0)


def test_last_positions_for_backward_trajectories_are_all_0():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(grid_dimension=grid_dim, grid_length=grid_length)
    trajectories, _, _ = runner.generate_backward_trajectories(batch_size)

    tf.debugging.assert_equal(trajectories[-1], 0)


def test_first_forward_actions_for_backward_trajectories_are_all_stop_actions():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(grid_dimension=grid_dim, grid_length=grid_length)
    _, _, forward_actions = runner.generate_backward_trajectories(batch_size)

    tf.debugging.assert_equal(forward_actions[0, :, :-1], 0)
    tf.debugging.assert_equal(forward_actions[0, :, -1], 1)


def test_last_backward_actions_for_backward_trajectories_are_all_0():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(grid_dimension=grid_dim, grid_length=grid_length)
    _, backward_actions, _ = runner.generate_backward_trajectories(batch_size)

    tf.debugging.assert_equal(backward_actions[-1], 0)


def test_actions_correctly_correspond_to_forward_trajectories():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(grid_dimension=grid_dim, grid_length=grid_length)
    trajectories, backward_actions, forward_actions = runner.generate_forward_trajectories(batch_size)

    next_forward_positions = trajectories[:-1] + forward_actions[:-1, :, :-1]
    next_backward_positions = trajectories[1:] - backward_actions[1:]

    tf.debugging.assert_equal(next_forward_positions, trajectories[1:])
    tf.debugging.assert_equal(next_backward_positions, trajectories[:-1])


def test_forward_trajectories_do_not_go_out_of_bounds():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(grid_dimension=grid_dim, grid_length=grid_length)
    trajectories, _, _ = runner.generate_forward_trajectories(batch_size)

    tf.debugging.assert_less_equal(trajectories, grid_length-1)


def test_first_positions_for_forward_trajectories_are_all_0():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(grid_dimension=grid_dim, grid_length=grid_length)
    trajectories, _, _ = runner.generate_forward_trajectories(batch_size)

    tf.debugging.assert_equal(trajectories[0], 0)


def test_first_backward_actions_for_forward_trajectories_are_all_0():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(grid_dimension=grid_dim, grid_length=grid_length)
    _, backward_actions, _ = runner.generate_forward_trajectories(batch_size)

    tf.debugging.assert_equal(backward_actions[0], 0)