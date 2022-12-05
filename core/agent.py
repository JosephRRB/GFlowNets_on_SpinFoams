import tensorflow as tf
from core.policy_network import PolicyNetwork


class Agent:
    NEG_INF = -10000000000.0

    def __init__(self, env_grid_dim, env_grid_length, exploration_rate=0.5):
        self.env_grid_dim = env_grid_dim
        self.env_grid_length = env_grid_length
        self.backward_action_dim = self.env_grid_dim
        self.forward_action_dim = self.env_grid_dim + 1

        self.exploration_rate = exploration_rate

        self.log_Z0 = tf.Variable(0.0, trainable=True, name="log_Z0")
        self.policy = PolicyNetwork(
            main_layer_nodes=[15],
            branch1_layer_nodes=[self.backward_action_dim],
            branch2_layer_nodes=[self.forward_action_dim],
        )
        self._build_policy_network()

    def _build_policy_network(self):
        # TODO: generalize later
        self.policy.main_layers[0].build(self.env_grid_length*self.env_grid_dim)
        self.policy.branch1_layers[0].build(15)
        self.policy.branch2_layers[0].build(15)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def act_backward(self, current_position):
        backward_action_logits, _ = self._get_action_logits(current_position)

        action_mask = self._find_forbidden_backward_actions(current_position)
        allowed_action_logits = self._mask_action_logits(backward_action_logits, action_mask)

        action_indices = self._choose_actions(allowed_action_logits)
        encoded_actions = self._encode_backward_actions(action_indices)

        is_still_sampling = self._check_if_able_to_act_backward(action_mask)
        backward_actions = encoded_actions * is_still_sampling
        return backward_actions

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.bool)
    ])
    def act_forward(self, current_position, is_still_sampling, training):
        _, forward_action_logits = self._get_action_logits(current_position)

        if training:
            uniform_noise = tf.random.uniform(shape=tf.shape(forward_action_logits), minval=1e-100)
            log_noise = tf.math.log(uniform_noise)
            forward_action_logits = (
                    self.exploration_rate * log_noise + (1 - self.exploration_rate) * forward_action_logits
            )

        action_mask = self._find_forbidden_forward_actions(current_position)
        allowed_action_logits = self._mask_action_logits(forward_action_logits, action_mask)

        action_indices = self._choose_actions(allowed_action_logits)
        encoded_actions = self._encode_forward_actions(action_indices)

        forward_actions = encoded_actions * is_still_sampling
        will_continue_to_sample = self._update_if_stop_action_is_chosen(is_still_sampling, forward_actions)
        return forward_actions, will_continue_to_sample

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)
    ])
    def calculate_action_log_probability_ratio(self, trajectories, backward_actions, forward_actions):
        backward_logits, forward_logits = self._get_action_logits_for_trajectories(trajectories)

        backward_log_probas = self._calculate_backward_action_log_probabilities(
            trajectories, backward_logits, backward_actions
        )
        forward_log_probas = self._calculate_forward_action_log_probabilities(
            trajectories, forward_logits, forward_actions
        )

        total_backward_log_probas = tf.reduce_sum(backward_log_probas, axis=0)
        total_forward_log_probas = tf.reduce_sum(forward_log_probas, axis=0)
        action_log_proba_ratios = total_forward_log_probas - total_backward_log_probas
        return action_log_proba_ratios

    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def _get_action_logits(self, current_position):
        encoded_position = self._encode_positions(current_position)
        backward_logits, forward_logits = self.policy(encoded_position)
        return backward_logits, forward_logits

    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def _encode_positions(self, position):
        encoded_position = tf.one_hot(position, depth=self.env_grid_length, axis=-1)
        return encoded_position

    @staticmethod
    # @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
    def _find_forbidden_backward_actions(position):
        backward_action_mask = tf.math.equal(position, 0)
        return backward_action_mask

    # @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
    def _find_forbidden_forward_actions(self, position):
        forward_action_mask = tf.math.equal(position, self.env_grid_length - 1)
        return forward_action_mask

    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.bool)
    # ])
    def _mask_action_logits(self, action_logits, mask):
        avoid_inds = tf.where(mask)
        orig_values = tf.gather_nd(action_logits, avoid_inds)
        add_values = -orig_values + self.NEG_INF
        # Need validation that masked are not sampled
        masked_logits = tf.tensor_scatter_nd_add(action_logits, avoid_inds, add_values)
        return masked_logits

    @staticmethod
    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32)])
    def _choose_actions(logits):
        action_indices = tf.random.categorical(logits, 1)
        return action_indices

    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.int64)])
    def _encode_backward_actions(self, action_indices):
        reshaped_action_indices = tf.reshape(action_indices, shape=(-1,))
        encoded_actions = tf.one_hot(reshaped_action_indices, depth=self.backward_action_dim, dtype=tf.int32)
        return encoded_actions

    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.int64)])
    def _encode_forward_actions(self, action_indices):
        reshaped_action_indices = tf.reshape(action_indices, shape=(-1,))
        encoded_actions = tf.one_hot(reshaped_action_indices, depth=self.forward_action_dim, dtype=tf.int32)
        return encoded_actions

    @staticmethod
    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.bool)])
    def _check_if_able_to_act_backward(backwards_action_mask):
        is_at_origin = tf.math.reduce_all(backwards_action_mask, axis=1, keepdims=True)
        is_able = tf.cast(tf.math.logical_not(is_at_origin), dtype=tf.int32)
        return is_able

    @staticmethod
    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, 1), dtype=tf.int32), tf.TensorSpec(shape=(None, None), dtype=tf.int32)
    # ])
    def _update_if_stop_action_is_chosen(still_sampling, forward_actions):
        will_continue_to_sample = still_sampling - tf.reshape(forward_actions[:, -1], shape=(-1, 1))
        return will_continue_to_sample

    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)])
    def _get_action_logits_for_trajectories(self, trajectories):
        shape = tf.shape(trajectories)
        max_traj_len = shape[0]
        batch_size = shape[1]

        reshaped_positions = tf.reshape(trajectories, shape=(-1, self.env_grid_dim))
        backward_logits, forward_logits = self._get_action_logits(reshaped_positions)
        trajectory_backward_logits = tf.reshape(
            backward_logits, shape=(max_traj_len, batch_size, self.backward_action_dim)
        )
        trajectory_forward_logits = tf.reshape(
            forward_logits, shape=(max_traj_len, batch_size, self.forward_action_dim)
        )
        return trajectory_backward_logits, trajectory_forward_logits

    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)
    # ])
    def _calculate_backward_action_log_probabilities(self, positions, logits, actions):
        action_mask = self._find_forbidden_backward_actions(positions)
        action_log_probas = self._calculate_log_probability_of_actions(action_mask, logits, actions)
        return action_log_probas

    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)
    # ])
    def _calculate_forward_action_log_probabilities(self, positions, logits, actions):
        action_mask = self._find_forbidden_forward_actions(positions)
        action_log_probas = self._calculate_log_probability_of_actions(action_mask, logits, actions)
        return action_log_probas

    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.bool),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)
    # ])
    def _calculate_log_probability_of_actions(self, action_mask, logits, actions):
        allowed_log_probas = self._normalize_allowed_action_logits(logits, action_mask)
        log_proba_of_chosen_actions = self._get_action_log_probas(allowed_log_probas, actions)
        return log_proba_of_chosen_actions

    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.bool),
    # ])
    def _normalize_allowed_action_logits(self, logits, mask):
        allowed_action_logits = self._mask_action_logits(logits, mask)
        allowed_log_probas = tf.nn.log_softmax(allowed_action_logits)
        return allowed_log_probas

    @staticmethod
    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
    # ])
    def _get_action_log_probas(log_probas, actions):
        action_log_probas = tf.reduce_sum(log_probas * tf.cast(actions, dtype=tf.float32), axis=2, keepdims=True)
        return action_log_probas
