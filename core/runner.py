import tensorflow as tf

from core.environment import HypergridEnvironment
from core.agent import Agent


class Runner:
    def __init__(self, grid_dimension, grid_length, learning_rate=0.0005):
        self.agent = Agent(
            env_grid_dim=grid_dimension, env_grid_length=grid_length
        )
        self.env = HypergridEnvironment(
            grid_dimension=grid_dimension, grid_length=grid_length
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_agent(self, batch_size, n_iterations):
        ave_losses = []
        for i in range(n_iterations):
            ave_loss = self._training_step(batch_size)
            ave_losses.append(ave_loss)

            if i % 100 == 0:
                print(f"Iteration: {i}, Average Loss: {float(ave_loss):.5f}")
        ave_losses = tf.concat(ave_losses, axis=0)
        return ave_losses

    def _training_step(self, batch_size):
        ts1, ba1, fa1 = self._generate_backward_trajectories(batch_size)
        ts2, ba2, fa2 = self._generate_forward_trajectories(batch_size, training=True)

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

    def _generate_backward_trajectories(self, batch_size):
        current_position = self.env.reset_for_backward_sampling(batch_size)

        no_coord_action = tf.zeros(
            shape=(batch_size, self.agent.backward_action_dim), dtype=tf.int32
        )

        trajectories = [current_position]
        backward_actions = []
        forward_actions = [no_coord_action]

        at_least_one_ongoing = tf.constant(True)
        while at_least_one_ongoing:
            action = self.agent.act_backward(current_position)
            current_position = self.env.step_backward(current_position, action)

            trajectories.append(current_position)
            backward_actions.append(action)
            forward_actions.append(action)

            at_least_one_ongoing = tf.math.reduce_any(
                tf.math.not_equal(current_position, 0)
            )
        backward_actions.append(no_coord_action)

        trajectory_max_len = len(trajectories)
        stop_actions = tf.scatter_nd(
            indices=[[0]],
            updates=tf.ones(shape=(1, batch_size, 1), dtype=tf.int32),
            shape=(trajectory_max_len, batch_size, 1)
        )
        trajectories = tf.stack(trajectories)
        backward_actions = tf.stack(backward_actions)
        forward_actions = tf.concat([tf.stack(forward_actions), stop_actions], axis=2)
        return trajectories, backward_actions, forward_actions

    def _generate_forward_trajectories(self, batch_size, training=False):
        current_position = self.env.reset_for_forward_sampling(batch_size)
        is_still_sampling = tf.ones(
            shape=(batch_size, 1), dtype=tf.int32
        )

        no_coord_action = tf.zeros(
            shape=(batch_size, self.agent.forward_action_dim), dtype=tf.int32
        )
        trajectories = [current_position]
        forward_actions = []
        backward_actions = [no_coord_action]

        at_least_one_ongoing = tf.constant(True)
        while at_least_one_ongoing:
            action, is_still_sampling = self.agent.act_forward(
                current_position, is_still_sampling, training=training
            )
            current_position = self.env.step_forward(current_position, action)

            trajectories.append(current_position)
            forward_actions.append(action)
            backward_actions.append(action)

            at_least_one_ongoing = tf.math.reduce_any(
                tf.math.equal(is_still_sampling, 1)
            )

        trajectories = tf.stack(trajectories[:-1])
        forward_actions = tf.stack(forward_actions)
        backward_actions = tf.stack(backward_actions[:-1])[:, :, :-1]
        return trajectories, backward_actions, forward_actions
