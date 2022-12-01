import tensorflow as tf


class Runner:
    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment

    def generate_backward_trajectories(self, batch_size):
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
        forward_actions = tf.concat([
            tf.stack(forward_actions),
            stop_actions
        ], axis=2)
        return trajectories, backward_actions, forward_actions

    def generate_forward_trajectories(self, batch_size):
        current_position = self.env.reset_for_forward_sampling(batch_size)
        is_still_sampling = tf.ones(
            shape=(batch_size, 1), dtype=tf.int32
        )
        trajectories = [current_position]
        actions = []

        at_least_one_ongoing = tf.constant(True)
        while at_least_one_ongoing:
            action, is_still_sampling = self.agent.act_forward(
                current_position, is_still_sampling
            )
            current_position = self.env.step_forward(current_position, action)
            trajectories.append(current_position)
            actions.append(action)

            at_least_one_ongoing = tf.math.reduce_any(
                tf.math.equal(is_still_sampling, 1)
            )

        trajectories = tf.stack(trajectories[:-1])

        forward_actions = tf.stack(actions)
        no_action = tf.zeros(
            shape=(batch_size, self.agent.forward_action_dim), dtype=tf.int32
        )
        backward_actions = tf.stack([no_action] + actions[:-1])[:, :, :-1]
        return trajectories, backward_actions, forward_actions
