import tensorflow as tf

from core.runner import Runner
from core.environment import SingleVertexSpinFoam


def test_actions_correctly_correspond_to_forward_trajectories():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))
    trajectories, backward_actions, forward_actions = runner._generate_forward_trajectories(batch_size, training=False)

    next_forward_positions = trajectories[:-1] + forward_actions[:-1, :, :-1]
    next_backward_positions = trajectories[1:] - backward_actions[1:]

    tf.debugging.assert_equal(next_forward_positions, trajectories[1:])
    tf.debugging.assert_equal(next_backward_positions, trajectories[:-1])


def test_forward_trajectories_do_not_go_out_of_bounds():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))
    trajectories, _, _ = runner._generate_forward_trajectories(batch_size, training=False)

    grid_length = int(2*spin_j+1)
    tf.debugging.assert_less_equal(trajectories, grid_length-1)


def test_first_positions_for_forward_trajectories_are_all_0():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))
    trajectories, _, _ = runner._generate_forward_trajectories(batch_size, training=False)

    tf.debugging.assert_equal(trajectories[0], 0)


def test_stop_action_correctly_stops_forward_trajectories():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))
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

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))
    _, backward_actions, _ = runner._generate_forward_trajectories(batch_size, training=False)

    tf.debugging.assert_equal(backward_actions[0], 0)


def test_actions_correctly_correspond_to_backward_trajectories():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))
    trajectories, backward_actions, forward_actions = runner._generate_backward_trajectories(batch_size)

    next_backward_positions = trajectories[:-1] - backward_actions[:-1]
    next_forward_positions = trajectories[1:] + forward_actions[1:, :, :-1]

    tf.debugging.assert_equal(trajectories[1:], next_backward_positions)
    tf.debugging.assert_equal(trajectories[:-1], next_forward_positions)


def test_backward_trajectories_do_not_go_out_of_bounds():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))
    trajectories, _, _ = runner._generate_backward_trajectories(batch_size)

    tf.debugging.assert_greater_equal(trajectories, 0)


def test_last_positions_for_backward_trajectories_are_all_0():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))
    trajectories, _, _ = runner._generate_backward_trajectories(batch_size)

    tf.debugging.assert_equal(trajectories[-1], 0)


def test_first_forward_actions_for_backward_trajectories_are_all_stop_actions():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))
    _, _, forward_actions = runner._generate_backward_trajectories(batch_size)

    tf.debugging.assert_equal(forward_actions[0, :, :-1], 0)
    tf.debugging.assert_equal(forward_actions[0, :, -1], 1)


def test_last_backward_actions_for_backward_trajectories_are_all_0():
    spin_j = 3.5
    batch_size = 10

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))
    _, backward_actions, _ = runner._generate_backward_trajectories(batch_size)

    tf.debugging.assert_equal(backward_actions[-1], 0)


def test_gradients_are_not_affected_by_padding_in_forward_trajectories():
    spin_j = 3.5

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))

    trajectory = tf.constant([
        [[0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 1]],
        [[0, 0, 0, 1, 1]],  # <- terminal state
        [[0, 0, 0, 1, 1]],  # <- padding
        [[0, 0, 0, 1, 1]],  # <- padding
    ])
    forward_actions = tf.constant([
        [[0, 0, 0, 0, 1, 0]],
        [[0, 0, 0, 1, 0, 0]],
        [[0, 0, 0, 0, 0, 1]],  # <- terminate action
        [[0, 0, 0, 0, 0, 0]],  # <- padding, no action
        [[0, 0, 0, 0, 0, 0]],  # <- padding, no action
    ])
    backward_actions = tf.constant([
        [[0, 0, 0, 0, 0]],  # <- no action, defined for grid origin
        [[0, 0, 0, 0, 1]],
        [[0, 0, 0, 1, 0]],
        [[0, 0, 0, 0, 0]],  # <- padding, no action
        [[0, 0, 0, 0, 0]],  # <- padding, no action
    ])

    dummy_log_Z0 = tf.constant(0.0, dtype=tf.float64)
    dummy_log_rewards = tf.zeros(shape=(trajectory.shape[0], 1), dtype=tf.float64)

    with tf.GradientTape() as tape_1:
        padded_log_proba_ratios = runner.agent.calculate_action_log_probability_ratio(
            trajectory, backward_actions, forward_actions
        )
        padded_ave_loss = runner._calculate_ave_loss(
            dummy_log_Z0, dummy_log_rewards, padded_log_proba_ratios
        )

    padded_grads = tape_1.gradient(
        padded_ave_loss, runner.agent.policy.trainable_weights
    )

    with tf.GradientTape() as tape_2:
        unpadded_log_proba_ratios = runner.agent.calculate_action_log_probability_ratio(
            trajectory[:3], backward_actions[:3], forward_actions[:3]
        )
        unpadded_ave_loss = runner._calculate_ave_loss(
            dummy_log_Z0, dummy_log_rewards, unpadded_log_proba_ratios
        )

    unpadded_grads = tape_2.gradient(
        unpadded_ave_loss, runner.agent.policy.trainable_weights
    )

    for pad, unpad in zip(padded_grads, unpadded_grads):
        tf.debugging.assert_near(pad, unpad)


def test_gradients_are_not_affected_by_padding_in_backward_trajectories():
    spin_j = 3.5

    runner = Runner(spinfoam_model=SingleVertexSpinFoam(spin_j=spin_j))

    trajectory = tf.constant([
        [[0, 0, 0, 2, 1]],
        [[0, 0, 0, 2, 0]],
        [[0, 0, 0, 1, 0]],
        [[0, 0, 0, 0, 0]],  # <- grid origin
        [[0, 0, 0, 0, 0]],  # <- padding
    ])
    backward_actions = tf.constant([
        [[0, 0, 0, 0, 1]],
        [[0, 0, 0, 1, 0]],
        [[0, 0, 0, 1, 0]],
        [[0, 0, 0, 0, 0]],  # <- no action, defined for grid origin
        [[0, 0, 0, 0, 0]],  # <- padding, no action
    ])
    forward_actions = tf.constant([
        [[0, 0, 0, 0, 0, 1]],  # <- terminate action, defined at start of backward trajectories
        [[0, 0, 0, 0, 1, 0]],
        [[0, 0, 0, 1, 0, 0]],
        [[0, 0, 0, 1, 0, 0]],
        [[0, 0, 0, 0, 0, 0]],  # <- padding, no action
    ])

    dummy_log_Z0 = tf.constant(0.0, dtype=tf.float64)
    dummy_log_rewards = tf.zeros(shape=(trajectory.shape[0], 1),
                                 dtype=tf.float64)

    with tf.GradientTape() as tape_1:
        padded_log_proba_ratios = runner.agent.calculate_action_log_probability_ratio(
            trajectory, backward_actions, forward_actions
        )
        padded_ave_loss = runner._calculate_ave_loss(
            dummy_log_Z0, dummy_log_rewards, padded_log_proba_ratios
        )

    padded_grads = tape_1.gradient(
        padded_ave_loss, runner.agent.policy.trainable_weights
    )

    with tf.GradientTape() as tape_2:
        unpadded_log_proba_ratios = runner.agent.calculate_action_log_probability_ratio(
            trajectory[:4], backward_actions[:4], forward_actions[:4]
        )
        unpadded_ave_loss = runner._calculate_ave_loss(
            dummy_log_Z0, dummy_log_rewards, unpadded_log_proba_ratios
        )

    unpadded_grads = tape_2.gradient(
        unpadded_ave_loss, runner.agent.policy.trainable_weights
    )

    for pad, unpad in zip(padded_grads, unpadded_grads):
        tf.debugging.assert_near(pad, unpad)
