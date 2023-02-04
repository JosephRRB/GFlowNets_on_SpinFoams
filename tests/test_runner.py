import tensorflow as tf

from core.runner import Runner

def test():
    # tf.config.run_functions_eagerly(True)
    runner = Runner(
        grid_length=8,
        main_layer_hidden_nodes=(30, 20),
        branch1_hidden_nodes=(),
        branch2_hidden_nodes=(),
        activation="swish",
        exploration_rate=0.5,
        learning_rate=0.01,
        environment_mode="spinfoam_vertex"
    )
    half_batch_size = tf.constant(500)
    n_iterations = tf.constant(10)

    evaluate_every_n_iterations = tf.constant(2)
    evaluation_batch_sizes = tf.constant([int(1e4), int(5e4), int(1e5), int(5e5)])

    ave_losses, distr_errors, agent_obss = runner.train_agent(
        half_batch_size, n_iterations, evaluate_every_n_iterations, evaluation_batch_sizes
    )


def test_actions_correctly_correspond_to_backward_trajectories():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(
        grid_dimension=grid_dim,
        grid_length=grid_length,
    )
    trajectories, backward_actions, forward_actions = runner._generate_backward_trajectories(batch_size)

    next_backward_positions = trajectories[:-1] - backward_actions[:-1]
    next_forward_positions = trajectories[1:] + forward_actions[1:, :, :-1]

    tf.debugging.assert_equal(trajectories[1:], next_backward_positions)
    tf.debugging.assert_equal(trajectories[:-1], next_forward_positions)


def test_backward_trajectories_do_not_go_out_of_bounds():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(
        grid_dimension=grid_dim,
        grid_length=grid_length,
    )
    trajectories, _, _ = runner._generate_backward_trajectories(batch_size)

    tf.debugging.assert_greater_equal(trajectories, 0)


def test_last_positions_for_backward_trajectories_are_all_0():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(
        grid_dimension=grid_dim,
        grid_length=grid_length,
    )
    trajectories, _, _ = runner._generate_backward_trajectories(batch_size)

    tf.debugging.assert_equal(trajectories[-1], 0)


def test_first_forward_actions_for_backward_trajectories_are_all_stop_actions():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(
        grid_dimension=grid_dim,
        grid_length=grid_length,
    )
    _, _, forward_actions = runner._generate_backward_trajectories(batch_size)

    tf.debugging.assert_equal(forward_actions[0, :, :-1], 0)
    tf.debugging.assert_equal(forward_actions[0, :, -1], 1)


def test_last_backward_actions_for_backward_trajectories_are_all_0():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(
        grid_dimension=grid_dim,
        grid_length=grid_length,
    )
    _, backward_actions, _ = runner._generate_backward_trajectories(batch_size)

    tf.debugging.assert_equal(backward_actions[-1], 0)


def test_actions_correctly_correspond_to_forward_trajectories():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(
        grid_dimension=grid_dim,
        grid_length=grid_length,
    )
    trajectories, backward_actions, forward_actions = runner._generate_forward_trajectories(batch_size, training=False)

    next_forward_positions = trajectories[:-1] + forward_actions[:-1, :, :-1]
    next_backward_positions = trajectories[1:] - backward_actions[1:]

    tf.debugging.assert_equal(next_forward_positions, trajectories[1:])
    tf.debugging.assert_equal(next_backward_positions, trajectories[:-1])


def test_forward_trajectories_do_not_go_out_of_bounds():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(
        grid_dimension=grid_dim,
        grid_length=grid_length,
    )
    trajectories, _, _ = runner._generate_forward_trajectories(batch_size, training=False)

    tf.debugging.assert_less_equal(trajectories, grid_length-1)


def test_first_positions_for_forward_trajectories_are_all_0():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(
        grid_dimension=grid_dim,
        grid_length=grid_length,
    )
    trajectories, _, _ = runner._generate_forward_trajectories(batch_size, training=False)

    tf.debugging.assert_equal(trajectories[0], 0)


def test_stop_action_correctly_stops_forward_trajectories():
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(
        grid_dimension=grid_dim,
        grid_length=grid_length,
    )
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
    grid_dim = 5
    grid_length = 8
    batch_size = 10

    runner = Runner(
        grid_dimension=grid_dim,
        grid_length=grid_length,
    )
    _, backward_actions, _ = runner._generate_forward_trajectories(batch_size, training=False)

    tf.debugging.assert_equal(backward_actions[0], 0)
