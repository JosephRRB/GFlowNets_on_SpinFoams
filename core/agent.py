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
        encoded_position = self._encode_position(current_position)
        backward_action_logits = self.policy.predict(encoded_position)[0]

        action_mask = self._find_forbidden_backward_actions(encoded_position)
        masked_logits = self._mask_action_logits(
            backward_action_logits, action_mask
        )

        action_indices = self._choose_action(masked_logits)
        encoded_actions = self._encode_backward_action(action_indices)

        is_still_sampling = self._check_if_able_to_act_backward(action_mask)
        backward_actions = encoded_actions * is_still_sampling
        return backward_actions

    def act_forward(self, current_position, is_still_sampling):
        encoded_position = self._encode_position(current_position)
        forward_action_logits = self.policy.predict(encoded_position)[1]

        action_mask = self._find_forbidden_forward_actions(encoded_position)
        masked_logits = self._mask_action_logits(
            forward_action_logits, action_mask
        )

        action_indices = self._choose_action(masked_logits)
        encoded_actions = self._encode_forward_action(action_indices)

        forward_actions = encoded_actions * is_still_sampling
        will_continue_to_sample = self._update_if_stop_action_is_chosen(
            is_still_sampling, forward_actions
        )
        return forward_actions, will_continue_to_sample


    def _encode_position(self, position):
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
            position, self.env_grid_length-1
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
    def _choose_action(logits):
        action_indices = tf.random.categorical(logits, 1)
        return action_indices

    def _encode_backward_action(self, action_indices):
        encoded_actions = tf.one_hot(
            tf.reshape(action_indices, shape=(-1,)),
            depth=self.backward_action_dim,
            dtype=tf.int32
        )
        return encoded_actions

    def _encode_forward_action(self, action_indices):
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

def _encode_state(state, intertwiner_dim):
    encoded = tf.reshape(
        tf.one_hot(state, intertwiner_dim, dtype=tf.float64),
        shape=(1, -1)
    )
    return encoded