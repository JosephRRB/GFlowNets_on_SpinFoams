import tensorflow as tf
from core.policy_network import PolicyNetwork


class Agent:
    NEG_INF = -10000000000.0

    def __init__(self, env_grid_dim, env_grid_length,
                 main_layer_hidden_nodes=(30, 20),
                 branch1_hidden_nodes=(10, ),
                 branch2_hidden_nodes=(10, ),
                 activation="swish",
                 exploration_rate=0.5
                 ):
        self.env_grid_dim = env_grid_dim
        self.env_grid_length = env_grid_length
        self.backward_action_dim = self.env_grid_dim
        self.forward_action_dim = self.env_grid_dim + 1
        self.exploration_rate = exploration_rate

        self.policy = PolicyNetwork(
            main_layer_nodes=[self.env_grid_length*self.env_grid_dim] + list(main_layer_hidden_nodes),
            branch1_layer_nodes=list(branch1_hidden_nodes) + [self.backward_action_dim],
            branch2_layer_nodes=list(branch2_hidden_nodes) + [self.forward_action_dim],
            activation=activation
        )
        self.log_Z0 = tf.Variable(
            0.0,
            trainable=True, name="log_Z0", dtype=tf.float64
        )

    def set_initial_estimate_for_log_z0(
            self, single_vertex_amplitudes, n_vertices, frac=0.99
    ):
        """
        Set a heuristic for the initial estimate of log_Z0. The aim is to shift
        the difference log_Z0 - log_rewards in the loss function to a smaller
        value.

        TODO: May be better to move the calculation in Runner()
        """
        abs_max = tf.reduce_max(tf.math.abs(single_vertex_amplitudes))
        estimate_log_max_reward = 2*n_vertices*tf.math.log(abs_max)
        self.log_Z0 = tf.Variable(
            frac*estimate_log_max_reward,
            trainable=True, name="log_Z0", dtype=tf.float64
        )

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.bool)
    ])
    def act_forward(self, current_position, is_still_sampling, training):
        """
        Select an action to generate a valid forward state transition. The
        current state is fed to the policy network which returns logits for the
        forward actions. The logits of invalid actions are then masked to avoid
        transitioning to states outside the environment grid. Based on the
        resulting masked logits, an action is chosen and encoded accordingly.
        Set the encoded action to zero (representing no action) if the
        trajectory has been flagged to be finished and update that trajectory
        flag if the terminate action was chosen for it.

        Parameters:
        -----------
        current_position:
            Batch of states shown to the agent. Based on these states, the agent
            will choose a valid action for each to transition them forward.

        is_still_sampling:
            Flag for each state in current_position. This represents whether the
            agent has terminated the generation of new states in a previous
            iteration. If is_still_sampling=0 for a specific state, the agent
            will not make an action on it so that it will remain the same for
            the next iterations.

        training:
            Boolean flag telling the agent whether it's generating trajectories
            for training or not. If training=True, noise is added to the action
            probabilities in order to encourage exploration of the environment
            grid. The amount of noise is governed by self.exploration_rate

        Return:
        -------
        forward_actions:
            Batch of one-hot encoded forward actions for each state in
            current_position. A 1 in the i^{th} index represents incrementing
            the i^{th} coordinate in a state by 1. The dimensionality is one
            more than the environment grid dimension so that a 1 in this last
            index represents the terminate action. If all are 0, this means that
            the agent will not change the state (this should only happen when
            the terminate action has been chosen in a previous iteration).

        will_continue_to_sample:
            Updated flags of is_still_sampling for the next iteration. If the
            terminate action is chosen, the flag will be updated from 1 to 0 and
            will remain 0 for the rest of the trajectory generation.
        """
        _, forward_action_logits = self._get_action_logits(current_position)
        if training:
            noise_per_action = tf.random.uniform(
                shape=tf.shape(forward_action_logits), dtype=tf.float64
            )
            noise_probs = noise_per_action / tf.math.reduce_sum(noise_per_action)
            forward_probs = tf.nn.softmax(forward_action_logits)
            noisy_probs = (
                self.exploration_rate * noise_probs +
                (1 - self.exploration_rate) * forward_probs
            )
            forward_action_logits = tf.math.log(noisy_probs)

        action_mask = self._find_forbidden_forward_actions(current_position)
        allowed_action_logits = self._mask_action_logits(forward_action_logits, action_mask)

        action_indices = self._choose_actions(allowed_action_logits)
        encoded_actions = self._encode_forward_actions(action_indices)

        forward_actions = encoded_actions * is_still_sampling
        will_continue_to_sample = self._update_if_stop_action_is_chosen(
            is_still_sampling, forward_actions
        )
        self._validate_forward_actions(
            forward_actions, action_mask, current_position, allowed_action_logits
        )
        return forward_actions, will_continue_to_sample

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.bool),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float64),
    ])
    def _validate_forward_actions(
        self, forward_actions, action_mask, current_position, allowed_action_logits
    ):
        invalid_action_row_inds = tf.where(
            tf.cast(action_mask, dtype=tf.int32)*forward_actions[:, :-1]
        )[:, 0]
        invalid_actions = tf.gather(forward_actions, invalid_action_row_inds)
        logits = tf.gather(allowed_action_logits, invalid_action_row_inds)
        positions = tf.gather(current_position, invalid_action_row_inds)
        tf.Assert(
            tf.math.equal(tf.shape(invalid_action_row_inds)[0], 0),
            [(
              f"agent chose actions={invalid_actions} "
              f"with logits={logits} "
              f"for states={positions} "
              f"which would be out of bounds"
            )]
        )

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)
    ])
    def calculate_action_log_probability_ratio(self, trajectories, backward_actions, forward_actions):
        """
        For each trajectory in the batch, calculate the logarithm of the product
        of ratios between forward and backward state transitions over the entire
        trajectory. Equivalently, calculate the difference between the total
        forward action log-probabilities and the total backward action
        log-probabilities for all trajectories in the batch.

        For each state in the batch of trajectories, get the forward and
        backward logits from the policy network. Mask the logits corresponding
        to invalid actions and normalize accordingly to produce
        log-probabilities. Use the actual one-hot encoded actions used to
        generate the trajectories to get their log-probabilities. For each
        trajectory, sum over all forward and backward log-probabilities
        separately and take the difference.

        Note that an artificial "no action" (which is all 0) is used by the
        agent to pad shorter trajectories. Thus, the padded states do not
        contribute to the sums. Additionally, the trajectories can be either
        generated by forward or backward actions. The sums are invariant with
        the direction used to create the trajectories.

        Parameters:
        -----------
        trajectories:
            - 3D Tensor of shape (max_traj_len, batch_size, grid_dim)
            - Forward or backward sequence of a batch of states. If generated
              from forward actions, sequences start from the grid origin and
              shorter trajectories are padded with their terminal states. Else,
              sequences start from random grid coordinates (treated as terminal
              states) which end in, and may be padded by, the grid origin.

        forward_actions:
            - 3D Tensor of shape (max_traj_len, batch_size, grid_dim + 1)
            - Sequence of a batch of one-hot encoded forward actions. If
              corresponding to forward trajectories, these are the actions used
              to generate those trajectories. Else, these are the opposite of
              the actions used to generate backward trajectories.

        backward_actions:
            - 3D Tensor of shape (max_traj_len, batch_size, grid_dim)
            - Sequence of a batch of one-hot encoded backward actions. If
              corresponding to forward trajectories, these are the opposite of
              the generating forward actions. Else, these are the actions used
              in generating those trajectories.

        Return:
        -------
        action_log_proba_ratios:
            - 2D Tensor of shape (batch_size, 1)
            - Total log probability ratios for actions throughout trajectories
              in the batch.
        """
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

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def _get_action_logits(self, current_position):
        """
        Get the logits corresponding to forward and backward actions by encoding
        the input batch of states and passing them to the policy network.
        """
        encoded_position = self._encode_positions(current_position)
        backward_logits, forward_logits = self.policy(encoded_position)
        return backward_logits, forward_logits

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def _encode_positions(self, position):
        """
        One-hot encode each coordinate of each state in the input batch.
        """
        encoded_position = tf.one_hot(
            position, depth=self.env_grid_length, axis=-1, dtype=tf.float64
        )
        return encoded_position

    @staticmethod
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
    def _find_forbidden_backward_actions(position):
        """
        Create a boolean mask with True specifying which coordinates cannot be
        decremented by a backward action. These correspond to the invalid
        backward actions for the input batch of states.
        """
        backward_action_mask = tf.math.equal(position, 0)
        return backward_action_mask

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
    def _find_forbidden_forward_actions(self, position):
        """
        Create a boolean mask with True specifying which coordinates cannot be
        incremented by a forward action. These correspond to the invalid
        forward actions for the input batch of states.
        """
        forward_action_mask = tf.math.equal(position, self.env_grid_length - 1)
        return forward_action_mask

    @tf.function(input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.bool)
    ])
    def _mask_action_logits(self, action_logits, mask):
        """
        Mask the logits corresponding to invalid actions by replacing the
        original values with a very large negative number (defined by
        self.NEG_INF). This should then give a zero probability for the invalid
        actions to be chosen.
        """
        avoid_inds = tf.where(mask)
        orig_values = tf.gather_nd(action_logits, avoid_inds)
        add_values = -orig_values + self.NEG_INF
        # Need validation that masked are not sampled
        masked_logits = tf.tensor_scatter_nd_add(action_logits, avoid_inds, add_values)
        return masked_logits

    @staticmethod
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float64)])
    def _choose_actions(logits):
        """
        Choose an action (represented by an index) based on the input batch of
        logits.
        """
        action_indices = tf.random.categorical(logits, 1)
        return action_indices

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.int64)])
    def _encode_forward_actions(self, action_indices):
        """
        One-hot encode the input batch of forward action indices.
        """
        reshaped_action_indices = tf.reshape(action_indices, shape=(-1,))
        encoded_actions = tf.one_hot(reshaped_action_indices, depth=self.forward_action_dim, dtype=tf.int32)
        return encoded_actions

    @staticmethod
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32)
    ])
    def _update_if_stop_action_is_chosen(still_sampling, forward_actions):
        """
        Update the still_sampling flags from 1 to 0 if the terminate action is
        chosen. Note that if still_sampling=0 already, then the corresponding
        forward_actions will be set to 0 ("no action") beforehand.
        """
        will_continue_to_sample = still_sampling - tf.reshape(forward_actions[:, -1], shape=(-1, 1))
        return will_continue_to_sample

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)])
    def _get_action_logits_for_trajectories(self, trajectories):
        """
        Wrapper of _get_action_logits() for a batch of trajectories. The
        sequence of states are reshaped into a 2D tensor first and are passed to
        _get_action_logits() to get the corresponding forward and backward
        logits for each state. These are then reshaped back accordingly to a 3D
        tensor.
        """
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

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float64),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)
    ])
    def _calculate_backward_action_log_probabilities(self, positions, logits, actions):
        """
        Wrapper of _calculate_log_probability_of_actions() for backward actions.
        """
        action_mask = self._find_forbidden_backward_actions(positions)
        action_log_probas = self._calculate_log_probability_of_actions(action_mask, logits, actions)
        return action_log_probas

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float64),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)
    ])
    def _calculate_forward_action_log_probabilities(self, positions, logits, actions):
        """
        Wrapper of _calculate_log_probability_of_actions() for forward actions.
        """
        action_mask = self._find_forbidden_forward_actions(positions)
        action_log_probas = self._calculate_log_probability_of_actions(action_mask, logits, actions)
        return action_log_probas

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.bool),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float64),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32)
    ])
    def _calculate_log_probability_of_actions(self, action_mask, logits, actions):
        """
        Calculate the log-probabilities for a batch of actions by masking out
        the logits corresponding to invalid actions, normalizing accordingly,
        and retrieving those associated with the input actions.
        """
        allowed_log_probas = self._normalize_allowed_action_logits(logits, action_mask)
        log_proba_of_chosen_actions = self._get_action_log_probas(allowed_log_probas, actions)
        return log_proba_of_chosen_actions

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float64),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.bool),
    ])
    def _normalize_allowed_action_logits(self, logits, mask):
        """
        Calculate log-probabilities by masking out logits corresponding to
        invalid actions and passing the result to a log_softmax function. Taking
        the exponential would then give probabilities for actions and
        specifically a zero probability for invalid actions.
        """
        allowed_action_logits = self._mask_action_logits(logits, mask)
        allowed_log_probas = tf.nn.log_softmax(allowed_action_logits)
        return allowed_log_probas

    @staticmethod
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float64),
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
    ])
    def _get_action_log_probas(log_probas, actions):
        """
        Get the log-probabilities for a batch of actions. Multiplying with the
        one-hot encoded actions would set all to zero except for the desired
        log-probabilities so that the sum retrieves those associated with the
        input actions. Specifically, the "no action" would give a zero which
        will not affect the total sum of log-probabilities for the entire
        trajectory.
        """
        action_log_probas = tf.reduce_sum(
            log_probas * tf.cast(actions, dtype=tf.float64),
            axis=2,
            keepdims=True
        )
        return action_log_probas


    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def act_backward(self, current_position):
        """
        Select an action to generate a valid backward state transition. The
        current state is fed to the policy network which returns logits for the
        backward actions. The logits of invalid actions are then masked to avoid
        transitioning to states outside the environment grid. Based on the
        resulting masked logits, an action is chosen and encoded accordingly.
        Set the encoded action to zero (representing no action) if the
        trajectory has reached the grid origin.

        Parameters:
        -----------
        current_position:
            Batch of states shown to the agent. Based on these states, the agent
            will choose a valid action for each to transition them backward.

        Return:
        -------
        backward_actions:
            Batch of one-hot encoded backward actions for each state in
            current_position. A 1 in the i^{th} index represents decrementing
            the i^{th} coordinate in a state by 1. If all are 0, this means that
            the agent will not change the state (this should only happen when
            the state is at the grid origin).
        """
        backward_action_logits, _ = self._get_action_logits(current_position)

        action_mask = self._find_forbidden_backward_actions(current_position)
        allowed_action_logits = self._mask_action_logits(backward_action_logits, action_mask)

        action_indices = self._choose_actions(allowed_action_logits)
        encoded_actions = self._encode_backward_actions(action_indices)

        is_still_sampling = self._check_if_able_to_act_backward(action_mask)
        backward_actions = encoded_actions * is_still_sampling
        return backward_actions

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.int64)])
    def _encode_backward_actions(self, action_indices):
        """
        One-hot encode the input batch of forward action indices.
        """
        reshaped_action_indices = tf.reshape(action_indices, shape=(-1,))
        encoded_actions = tf.one_hot(reshaped_action_indices,
                                     depth=self.backward_action_dim,
                                     dtype=tf.int32)
        return encoded_actions

    @staticmethod
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.bool)])
    def _check_if_able_to_act_backward(backwards_action_mask):
        """
        Create a flag for each state in the batch representing whether the state
        is at the grid origin so that there is no longer any valid backward
        action. If is_able=0 for a specific state, the agent will not make an
        action on it so that it will remain at the grid origin for the next
        iterations.

        The flag is_able is similar to is_still_sampling in act_forward()
        """
        is_at_origin = tf.math.reduce_all(backwards_action_mask, axis=1,
                                          keepdims=True)
        is_able = tf.cast(tf.math.logical_not(is_at_origin), dtype=tf.int32)
        return is_able