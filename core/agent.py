import tensorflow as tf
from core.policy_network import PolicyNetwork


class Agent:
    NEG_INF = -10000000000.0

    def __init__(self, env_grid_dim, env_grid_length):
        self.env_grid_dim = env_grid_dim
        self.env_grid_length = env_grid_length
        self.backward_action_dim = self.env_grid_dim
        self.forward_action_dim = self.env_grid_dim + 1

        self.policy = PolicyNetwork(
            main_layer_nodes=[15],
            branch1_layer_nodes=[self.backward_action_dim],
            branch2_layer_nodes=[self.forward_action_dim],
        )

    def act_backward(self, current_position):
        encoded_position = self._encode_positions(current_position)
        backward_action_logits = self.policy.predict(encoded_position)[0]

        action_mask = self._find_forbidden_backward_actions(encoded_position)
        allowed_action_logits = self._mask_action_logits(
            backward_action_logits, action_mask
        )

        action_indices = self._choose_actions(allowed_action_logits)
        encoded_actions = self._encode_backward_actions(action_indices)

        is_still_sampling = self._check_if_able_to_act_backward(action_mask)
        backward_actions = encoded_actions * is_still_sampling
        return backward_actions

    def act_forward(self, current_position, is_still_sampling):
        encoded_position = self._encode_positions(current_position)
        forward_action_logits = self.policy.predict(encoded_position)[1]

        action_mask = self._find_forbidden_forward_actions(encoded_position)
        allowed_action_logits = self._mask_action_logits(
            forward_action_logits, action_mask
        )

        action_indices = self._choose_actions(allowed_action_logits)
        encoded_actions = self._encode_forward_actions(action_indices)

        forward_actions = encoded_actions * is_still_sampling
        will_continue_to_sample = self._update_if_stop_action_is_chosen(
            is_still_sampling, forward_actions
        )
        return forward_actions, will_continue_to_sample

    def calculate_action_log_probability_ratio(
            self, trajectories, backward_actions, forward_actions
    ):
        reshaped_positions = tf.reshape(
            trajectories, shape=(-1, self.env_grid_dim)
        )
        encoded_positions = self._encode_positions(reshaped_positions)
        backward_logits, forward_logits = self.policy.predict(encoded_positions)

        reshaped_backward_logits = tf.reshape(
            backward_logits, shape=backward_actions.shape
        )
        backward_log_probas = self._calculate_backward_action_log_probabilities(
            trajectories, reshaped_backward_logits, backward_actions
        )

        reshaped_forward_logits = tf.reshape(
            forward_logits, shape=forward_actions.shape
        )
        forward_log_probas = self._calculate_forward_action_log_probabilities(
            trajectories, reshaped_forward_logits, forward_actions
        )

        total_backward_log_probas = tf.reduce_sum(backward_log_probas, axis=0)
        total_forward_log_probas = tf.reduce_sum(forward_log_probas, axis=0)
        action_log_proba_ratios = total_forward_log_probas - total_backward_log_probas
        return action_log_proba_ratios

    def _encode_positions(self, position):
        encoded_position = tf.one_hot(
            position, depth=self.env_grid_length, axis=-1
        )
        return encoded_position

    @staticmethod
    def _find_forbidden_backward_actions(position):
        backward_action_mask = tf.math.equal(position, 0)
        return backward_action_mask

    def _find_forbidden_forward_actions(self, position):
        forward_action_mask = tf.math.equal(
            position, self.env_grid_length - 1
        )
        return forward_action_mask

    def _mask_action_logits(self, action_logits, mask):
        avoid_inds = tf.where(mask)
        # Need validation that masked are not sampled
        masked_logits = tf.tensor_scatter_nd_add(
            action_logits, avoid_inds,
            tf.constant([self.NEG_INF] * avoid_inds.shape[0])
        )
        return masked_logits

    @staticmethod
    def _choose_actions(logits):
        action_indices = tf.random.categorical(logits, 1)
        return action_indices

    def _encode_backward_actions(self, action_indices):
        encoded_actions = tf.one_hot(
            tf.reshape(action_indices, shape=(-1,)),
            depth=self.backward_action_dim,
            dtype=tf.int32
        )
        return encoded_actions

    def _encode_forward_actions(self, action_indices):
        encoded_actions = tf.one_hot(
            tf.reshape(action_indices, shape=(-1,)),
            depth=self.forward_action_dim,
            dtype=tf.int32
        )
        return encoded_actions

    @staticmethod
    def _check_if_able_to_act_backward(backwards_action_mask):
        is_at_origin = tf.math.reduce_all(
            backwards_action_mask, axis=1, keepdims=True
        )
        is_able = tf.cast(
            tf.math.logical_not(is_at_origin),
            dtype=tf.int32
        )
        return is_able

    @staticmethod
    def _update_if_stop_action_is_chosen(still_sampling, forward_actions):
        will_continue_to_sample = (
                still_sampling - tf.reshape(forward_actions[:, -1], shape=(-1, 1))
        )
        return will_continue_to_sample

    def _calculate_backward_action_log_probabilities(
            self, positions, logits, actions
    ):
        action_mask = self._find_forbidden_backward_actions(positions)
        action_log_probas = self._calculate_log_probability_of_actions(
            action_mask, logits, actions
        )
        return action_log_probas

    def _calculate_forward_action_log_probabilities(
            self, positions, logits, actions
    ):
        action_mask = self._find_forbidden_forward_actions(positions)
        action_log_probas = self._calculate_log_probability_of_actions(
            action_mask, logits, actions
        )
        return action_log_probas

    def _calculate_log_probability_of_actions(
            self, action_mask, logits, actions
    ):
        allowed_log_probas = self._normalize_allowed_action_logits(
            logits, action_mask
        )
        log_proba_of_chosen_actions = self._get_action_log_probas(
            allowed_log_probas, actions
        )
        return log_proba_of_chosen_actions

    def _normalize_allowed_action_logits(self, logits, mask):
        allowed_action_logits = self._mask_action_logits(logits, mask)
        allowed_log_probas = tf.nn.log_softmax(allowed_action_logits)
        return allowed_log_probas

    @staticmethod
    def _get_action_log_probas(log_probas, actions):
        action_log_probas = tf.reduce_sum(
            log_probas * tf.cast(actions, dtype=tf.float32),
            axis=2, keepdims=True
        )
        return action_log_probas
