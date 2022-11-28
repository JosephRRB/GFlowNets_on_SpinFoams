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

    def act_backwards(self, current_position):
        encoded_position = self._encode_position(current_position)
        backward_action_logits = self.policy.predict(encoded_position)[0]

        action_mask = self._find_forbidden_backward_actions(encoded_position)
        masked_logits = self._mask_action_logits(
            backward_action_logits, action_mask
        )

        action_indices = self._choose_action(masked_logits)
        encoded_actions = self._encode_backward_action(action_indices)

        is_still_sampling = self._check_if_able_to_act_backwards(action_mask)
        backward_actions = encoded_actions*is_still_sampling
        return backward_actions

    @staticmethod
    def _choose_action(logits):
        action_indices = tf.random.categorical(logits, 1)
        return action_indices

    def _encode_position(self, position):
        encoded_position = tf.one_hot(
            position, depth=self.env_grid_length, axis=-1
        )
        return encoded_position

    def _encode_backward_action(self, action_indices):
        encoded_actions = tf.one_hot(
            tf.reshape(action_indices, shape=(-1,)),
            depth=self.backward_action_dim,
            dtype=tf.int32
        )
        return encoded_actions

    @staticmethod
    def _find_forbidden_backward_actions(position):
        backwards_action_mask = tf.math.equal(position, 0)
        return backwards_action_mask

    def _check_if_able_to_act_backwards(self, backwards_action_mask):
        is_at_origin = tf.math.reduce_all(
            backwards_action_mask, axis=1, keepdims=True
        )
        is_able = tf.cast(
            tf.math.logical_not(is_at_origin),
            dtype=tf.int32
        )
        return is_able

    def _mask_action_logits(self, action_logits, mask):
        avoid_inds = tf.where(mask)
        # Need validation that masked are not sampled
        masked_logits = tf.tensor_scatter_nd_add(
            action_logits, avoid_inds,
            tf.constant([self.NEG_INF] * avoid_inds.shape[0])
        )
        return masked_logits

def _encode_state(state, intertwiner_dim):
    encoded = tf.reshape(
        tf.one_hot(state, intertwiner_dim, dtype=tf.float64),
        shape=(1, -1)
    )
    return encoded