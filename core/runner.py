import tensorflow as tf

from core.environment import HypergridEnvironment
from core.agent import Agent


class Runner:
    def __init__(self, grid_dimension, grid_length,
                 main_layer_hidden_nodes, branch1_hidden_nodes, branch2_hidden_nodes,
                 exploration_rate=0.5,
                 learning_rate=0.0005
                 ):
        self.agent = Agent(
            grid_dimension, grid_length,
            main_layer_hidden_nodes, branch1_hidden_nodes, branch2_hidden_nodes,
            exploration_rate=exploration_rate
        )
        self.env = HypergridEnvironment(
            grid_dimension=grid_dimension, grid_length=grid_length
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_agent(self, batch_size, n_iterations, check_loss_every_n_iterations):
        ave_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(n_iterations):
            ave_loss = self._training_step(batch_size)
            ave_losses = ave_losses.write(i, ave_loss)

            if tf.math.equal(tf.math.floormod(i, check_loss_every_n_iterations), 0):
                tf.print("Iteration:",  i, "Average Loss:", ave_loss)
        ave_losses = ave_losses.stack()
        return ave_losses

    @tf.function
    def generate_samples_from_agent(self, batch_size):
        current_position = self.env.reset_for_forward_sampling(batch_size)
        is_still_sampling = tf.ones(
            shape=(batch_size, 1), dtype=tf.int32
        )
        training = tf.constant(False)

        at_least_one_ongoing = tf.constant(True)
        while at_least_one_ongoing:
            action, is_still_sampling = self.agent.act_forward(
                current_position, is_still_sampling, training
            )
            current_position = self.env.step_forward(current_position, action)
            at_least_one_ongoing = tf.math.reduce_any(
                tf.math.equal(is_still_sampling, 1)
            )
        return current_position

    def _get_normalized_agent_sample_distribution(self, batch_size):
        samples = self.generate_samples_from_agent(batch_size)
        sample_counts = self._count_sampled_grid_coordinates(samples)

        # normalized_env_rewards = self.env.rewards / tf.math.reduce_sum(self.env.rewards)
        normalized_agent_sample_distribution = sample_counts / tf.math.reduce_sum(sample_counts)

        # abs_error = tf.math.abs(normalized_agent_sample_distribution - normalized_env_rewards)
        # return abs_error, normalized_agent_sample_distribution, normalized_env_rewards
        return normalized_agent_sample_distribution

    @tf.function
    def _count_sampled_grid_coordinates(self, samples):
        n_samples = tf.shape(samples)[0]
        zeros = tf.zeros(shape=[self.env.grid_length] * self.env.grid_dimension, dtype=tf.float32)
        updates = tf.ones(shape=[n_samples], dtype=tf.float32)
        counts = tf.tensor_scatter_nd_add(zeros, samples, updates)
        return counts

    @tf.function
    def _training_step(self, batch_size):
        ts1, ba1, fa1 = self._generate_backward_trajectories(batch_size)
        ts2, ba2, fa2 = self._generate_forward_trajectories(batch_size, training=tf.constant(True))

        final_positions = tf.concat([ts1[0], ts2[-1]], axis=0)
        rewards = self.env.get_rewards(final_positions)
        with tf.GradientTape() as tape:
            alpr1 = self.agent.calculate_action_log_probability_ratio(ts1, ba1, fa1)
            alpr2 = self.agent.calculate_action_log_probability_ratio(ts2, ba2, fa2)
            action_log_proba_ratios = tf.concat([alpr1, alpr2], axis=0)
            log_Z0 = self.agent.log_Z0
            log_rewards = tf.math.log(rewards)
            ave_loss = self._calculate_ave_loss(log_Z0, log_rewards, action_log_proba_ratios)

        grads = tape.gradient(
            ave_loss, self.agent.policy.trainable_weights + [self.agent.log_Z0]
        )
        self.optimizer.apply_gradients(
            zip(grads, self.agent.policy.trainable_weights + [self.agent.log_Z0])
        )
        return ave_loss


    @staticmethod
    @tf.function
    def _calculate_ave_loss(log_Z0, log_rewards, action_log_proba_ratios):
        loss = tf.math.square(log_Z0 - log_rewards + action_log_proba_ratios)
        ave_loss = tf.reduce_mean(loss)
        return ave_loss

    @tf.function
    def _generate_backward_trajectories(self, batch_size):
        current_position = self.env.reset_for_backward_sampling(batch_size)

        no_coord_action = tf.zeros(
            shape=(batch_size, self.agent.backward_action_dim), dtype=tf.int32
        )

        trajectories = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        backward_actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        forward_actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        trajectories = trajectories.write(0, current_position)
        forward_actions = forward_actions.write(0, no_coord_action)

        i = tf.constant(0)
        at_least_one_ongoing = tf.constant(True)
        while at_least_one_ongoing:
            action = self.agent.act_backward(current_position)
            current_position = self.env.step_backward(current_position, action)

            trajectories = trajectories.write(i+1, current_position)
            backward_actions = backward_actions.write(i, action)
            forward_actions = forward_actions.write(i+1, action)

            i += 1
            at_least_one_ongoing = tf.math.reduce_any(
                tf.math.not_equal(current_position, 0)
            )
        backward_actions = backward_actions.write(i, no_coord_action)

        trajectory_max_len = i + 1
        stop_actions = tf.scatter_nd(
            indices=[[0]],
            updates=tf.ones(shape=(1, batch_size, 1), dtype=tf.int32),
            shape=(trajectory_max_len, batch_size, 1)
        )
        trajectories = trajectories.stack()
        backward_actions = backward_actions.stack()
        forward_actions = tf.concat([forward_actions.stack(), stop_actions], axis=2)
        return trajectories, backward_actions, forward_actions

    @tf.function
    def _generate_forward_trajectories(self, batch_size, training):
        current_position = self.env.reset_for_forward_sampling(batch_size)
        is_still_sampling = tf.ones(
            shape=(batch_size, 1), dtype=tf.int32
        )

        no_coord_action = tf.zeros(
            shape=(batch_size, self.agent.forward_action_dim), dtype=tf.int32
        )

        trajectories = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        backward_actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        forward_actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        trajectories = trajectories.write(0, current_position)
        backward_actions = backward_actions.write(0, no_coord_action)

        i = tf.constant(0)
        at_least_one_ongoing = tf.constant(True)
        while at_least_one_ongoing:
            action, is_still_sampling = self.agent.act_forward(
                current_position, is_still_sampling, training
            )
            current_position = self.env.step_forward(current_position, action)

            trajectories = trajectories.write(i + 1, current_position)
            backward_actions = backward_actions.write(i + 1, action)
            forward_actions = forward_actions.write(i, action)

            i += 1
            at_least_one_ongoing = tf.math.reduce_any(
                tf.math.equal(is_still_sampling, 1)
            )

        trajectories = trajectories.stack()[:-1]
        forward_actions = forward_actions.stack()
        backward_actions = backward_actions.stack()[:-1, :, :-1]
        return trajectories, backward_actions, forward_actions
