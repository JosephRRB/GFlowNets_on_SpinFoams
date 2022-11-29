import tensorflow as tf
from core.agent import Agent


def test_position_is_correctly_encoded():
    grid_dim = 5
    grid_length = 8
    agent = Agent(
        env_grid_dim=grid_dim,
        env_grid_length=grid_length
    )
    position = tf.constant([
        [6, 3, 5, 0, 3],
        [1, 7, 4, 2, 0],
    ], dtype=tf.int32)
    encoded_position = agent._encode_positions(position)

    expected = tf.constant([
        [
            [0, 0, 0, 0, 0, 0, 1, 0],  # 6
            [0, 0, 0, 1, 0, 0, 0, 0],  # 3
            [0, 0, 0, 0, 0, 1, 0, 0],  # 5
            [1, 0, 0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 1, 0, 0, 0, 0],  # 3
        ],
        [
            [0, 1, 0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 0, 0, 0, 0, 1],  # 7
            [0, 0, 0, 0, 1, 0, 0, 0],  # 4
            [0, 0, 1, 0, 0, 0, 0, 0],  # 2
            [1, 0, 0, 0, 0, 0, 0, 0],  # 0
        ],
    ], dtype=tf.float32)

    assert encoded_position.shape == (2, grid_dim, grid_length)
    tf.debugging.assert_equal(encoded_position, expected)


def test_forbidden_backward_actions_are_correct():
    position = tf.constant([
        [6, 3, 5, 0, 3],
        [1, 7, 4, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=tf.int32)
    action_mask = Agent._find_forbidden_backward_actions(position)

    expected = tf.constant([
        [False, False, False, True, False],
        [False, False, False, False, True],
        [True, True, True, True, True],
    ])

    assert action_mask.shape == position.shape
    tf.debugging.assert_equal(action_mask, expected)


def test_forbidden_forward_actions_are_correct():
    grid_dim = 5
    grid_length = 8
    agent = Agent(
        env_grid_dim=grid_dim,
        env_grid_length=grid_length
    )
    position = tf.constant([
        [6, 3, 5, 0, 3],
        [7, 7, 7, 7, 7],
        [1, 7, 4, 7, 0],
    ], dtype=tf.int32)
    action_mask = agent._find_forbidden_forward_actions(position)

    expected = tf.constant([
        [False, False, False, False, False],
        [True, True, True, True, True],
        [False, True, False, True, False],
    ])

    assert action_mask.shape == position.shape
    tf.debugging.assert_equal(action_mask, expected)


def test_action_logits_are_correctly_masked():
    grid_dim = 5
    grid_length = 8
    agent = Agent(
        env_grid_dim=grid_dim,
        env_grid_length=grid_length
    )
    action_logits = tf.constant([
        [-agent.NEG_INF, -3.2, 0.4, -7.1, 5.8, -0.8],
        [-3.6, -5.1, -2.7, 4.5, 6.1, 3.2],
    ])
    mask = tf.constant([
        [True, True, False, True, True],
        [False, True, False, True, False],
    ])

    masked_action_logits = agent._mask_action_logits(
        action_logits, mask
    )

    forbidden_inds = tf.where(mask)
    allowed_inds = tf.where(tf.math.logical_not(mask))
    forbidden_action_logits = tf.gather_nd(masked_action_logits, forbidden_inds)
    allowed_action_logits = tf.gather_nd(masked_action_logits, allowed_inds)
    original_allowed_action_logits = tf.gather_nd(action_logits, allowed_inds)

    assert masked_action_logits.shape == action_logits.shape
    tf.debugging.assert_equal(forbidden_action_logits, agent.NEG_INF)
    tf.debugging.assert_equal(allowed_action_logits, original_allowed_action_logits)


def test_forbidden_actions_are_not_chosen():
    # This test is not strict
    neg_inf = Agent.NEG_INF
    masked_logits = tf.constant(
        [[neg_inf, neg_inf, neg_inf, -3.1, neg_inf]] * 1000 +
        [[1.3, -4.2, 3.4, neg_inf, -0.7]] * 1000
    )
    # Note: If all elements in a row are neg_inf, _choose_actions will still give
    # a "chosen index". But this will be removed downstream by is_still_sampling
    action_indices = Agent._choose_actions(masked_logits)

    assert action_indices.shape == (masked_logits.shape[0], 1)
    assert set(action_indices[:1000, 0].numpy()) == {3}
    assert 3 not in set(action_indices[1000:, 0].numpy())


def test_backward_actions_are_correctly_encoded():
    grid_dim = 5
    grid_length = 8
    agent = Agent(
        env_grid_dim=grid_dim,
        env_grid_length=grid_length
    )
    action_indices = tf.constant([
        [0], [1], [2], [3], [4]
    ])
    encoded_actions = agent._encode_backward_actions(action_indices)

    expected = tf.constant([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])

    assert encoded_actions.shape == (action_indices.shape[0], agent.backward_action_dim)
    tf.debugging.assert_equal(encoded_actions, expected)


def test_forward_actions_are_correctly_encoded():
    grid_dim = 5
    grid_length = 8
    agent = Agent(
        env_grid_dim=grid_dim,
        env_grid_length=grid_length
    )
    action_indices = tf.constant([
        [0], [1], [2], [3], [4], [5]
    ])
    encoded_actions = agent._encode_forward_actions(action_indices)

    expected = tf.constant([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])

    assert encoded_actions.shape == (action_indices.shape[0], agent.forward_action_dim)
    tf.debugging.assert_equal(encoded_actions, expected)


def test_still_sampling_flag_is_0_if_no_available_backward_action():
    action_mask = tf.constant([
        [True, True, True, True, True],
        [False, False, False, False, False],
        [False, True, False, True, False],
    ])
    is_still_sampling = Agent._check_if_able_to_act_backward(action_mask)
    expected = tf.constant([
        [0], [1], [1]
    ])

    assert is_still_sampling.shape == (action_mask.shape[0], 1)
    tf.debugging.assert_equal(is_still_sampling, expected)


def test_still_sampling_flag_becomes_0_if_stop_action_is_chosen():
    forward_actions = tf.constant([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    is_still_sampling = tf.constant([
        [1], [1], [1], [1], [1], [1],
    ])
    will_continue_to_sample = Agent._update_if_stop_action_is_chosen(
        is_still_sampling, forward_actions
    )

    expected = tf.constant([
        [1], [1], [1], [1], [1], [0],
    ])

    assert will_continue_to_sample.shape == is_still_sampling.shape
    assert will_continue_to_sample.shape == (forward_actions.shape[0], 1)
    tf.debugging.assert_equal(will_continue_to_sample, expected)
