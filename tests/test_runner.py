import tensorflow as tf

from core.runner import Runner


def test_actions_correctly_correspond_to_forward_trajectories():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spin_j=spin_j)
    trajectories, backward_actions, forward_actions = runner._generate_forward_trajectories(batch_size, training=False)

    next_forward_positions = trajectories[:-1] + forward_actions[:-1, :, :-1]
    next_backward_positions = trajectories[1:] - backward_actions[1:]

    tf.debugging.assert_equal(next_forward_positions, trajectories[1:])
    tf.debugging.assert_equal(next_backward_positions, trajectories[:-1])


def test_forward_trajectories_do_not_go_out_of_bounds():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spin_j=spin_j)
    trajectories, _, _ = runner._generate_forward_trajectories(batch_size, training=False)

    grid_length = int(2*spin_j+1)
    tf.debugging.assert_less_equal(trajectories, grid_length-1)


def test_first_positions_for_forward_trajectories_are_all_0():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spin_j=spin_j)
    trajectories, _, _ = runner._generate_forward_trajectories(batch_size, training=False)

    tf.debugging.assert_equal(trajectories[0], 0)


def test_stop_action_correctly_stops_forward_trajectories():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spin_j=spin_j)
    trajectories, _, forward_actions = runner._generate_forward_trajectories(batch_size, training=False)

    stop_actions = forward_actions[:, :, -1]
    stop_action_indices = tf.where(stop_actions)
    coord_when_stop_was_chosen = tf.gather_nd(trajectories, stop_action_indices)
    corresponding_last_coords = tf.gather_nd(
        trajectories[-1, :, :], tf.reshape(stop_action_indices[:, 1], shape=(-1, 1))
    )

    tf.debugging.assert_equal(coord_when_stop_was_chosen, corresponding_last_coords)
    tf.debugging.assert_equal(tf.reduce_sum(stop_actions, axis=0), 1)


def test_first_backward_actions_for_forward_trajectories_are_all_0():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spin_j=spin_j)
    _, backward_actions, _ = runner._generate_forward_trajectories(batch_size, training=False)

    tf.debugging.assert_equal(backward_actions[0], 0)
