import tensorflow as tf

from core.environment import HypergridEnvironment, _calculate_dihedral_angles
from core.agent import Agent

class Runner:
    def __init__(self,
                 main_layer_hidden_nodes=(30, 20),
                 branch1_hidden_nodes=(10, ),
                 branch2_hidden_nodes=(10, ),
                 activation="swish",
                 exploration_rate=0.5,
                 grid_dimension=5,
                 grid_length=8,
                 environment_mode="test_grid",
                 learning_rate=0.0005
                 ):
        if environment_mode == "spinfoam_vertex":
            grid_dimension = 5

        self.agent = Agent(
            grid_dimension, grid_length,
            main_layer_hidden_nodes=main_layer_hidden_nodes,
            branch1_hidden_nodes=branch1_hidden_nodes,
            branch2_hidden_nodes=branch2_hidden_nodes,
            activation=activation,
            exploration_rate=exploration_rate
        )
        self.env = HypergridEnvironment(
            grid_dimension=grid_dimension, grid_length=grid_length,
            environment_mode=environment_mode,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
    ])
    def train_agent(self, half_batch_size, n_iterations, evaluate_every_n_iterations, evaluation_batch_size):
        ave_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        distr_js_dists = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        observables = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        samples = tf.TensorArray(dtype=tf.int8, size=0, dynamic_size=True)

        # distr_errors, obss = self.evaluate_agent_on_batches(evaluation_batch_sizes)
        distr_js_dist, obs = self.evaluate_agent(evaluation_batch_size)
        distr_js_dists = distr_js_dists.write(0, distr_js_dist)
        observables = observables.write(0, obs)

        eval_i = 1
        for i in tf.range(n_iterations):
            ave_loss = self._training_step(half_batch_size)
            ave_losses = ave_losses.write(i, ave_loss)

            if tf.math.equal(tf.math.floormod(i+1, evaluate_every_n_iterations), 0):
                tf.print("Nth iteration:",  i+1, "Average Loss:", ave_loss)
                # agent_observable = self.calculate_observable_from_agent(evaluation_batch_size)
                # distr_errors, obss = self.evaluate_agent_on_batches(evaluation_batch_sizes)
                distr_js_dist, obs = self.evaluate_agent(evaluation_batch_size)
                distr_js_dists = distr_js_dists.write(eval_i, distr_js_dist)
                observables = observables.write(eval_i, obs)
                eval_i += 1

        ave_losses = ave_losses.stack()
        distr_js_dists = distr_js_dists.stack()
        observables = observables.stack()
        samples = samples.stack()
        
        return ave_losses, distr_js_dists, observables

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
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

    # @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
    # def get_normalized_agent_sample_distribution(self, batch_size):
    #     samples = self.generate_samples_from_agent(batch_size)
    #     sample_counts = self._count_sampled_grid_coordinates(samples)
    #
    #     normalized_agent_sample_distribution = sample_counts / tf.math.reduce_sum(sample_counts)
    #     return normalized_agent_sample_distribution

    def evaluate_agent(self, batch_size):
        samples = self.generate_samples_from_agent(batch_size)
        
        print(samples.numpy())

        sample_counts = self._count_sampled_grid_coordinates(samples)
        agent_distr = sample_counts / tf.math.reduce_sum(sample_counts)
        # distr_ave_l1_error = tf.math.reduce_mean(tf.abs(agent_distr - self.env.rewards))
        distr_js_dist = _compute_js_dist(
            tf.reshape(agent_distr, shape=(-1,)),
            tf.reshape(self.env.rewards, shape=(-1,))
        )

        i1s = tf.cast(samples[:, 0], dtype=tf.float32)
        agent_observable = tf.math.reduce_mean(
            _calculate_dihedral_angles(i1s, self.env.spin_j)
        )
        # observable_l1_error = tf.abs(
        #     agent_ave_dihedral_angle - self.env.theoretical_ave_dihedral_angle
        # )
        return distr_js_dist, agent_observable

    # def calculate_observable_from_agent(self, batch_size):
    #     samples = self.generate_samples_from_agent(batch_size)
    #     i1s = tf.cast(samples[:, 0], dtype=tf.float32)
    #     agent_observable = tf.math.reduce_mean(
    #         _calculate_dihedral_angles(i1s, self.env.spin_j)
    #     )
    #     return agent_observable

    # def evaluate_agent_on_batches(self, batch_sizes):
    #     distr_errors = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #     observables = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    #
    #     ind = 0
    #     for batch_size in batch_sizes:
    #         distr_ave_l1_error, agent_observable = self.evaluate_agent(batch_size)
    #         distr_errors = distr_errors.write(ind, distr_ave_l1_error)
    #         observables = observables.write(ind, agent_observable)
    #         ind += 1
    #
    #     distr_errors = distr_errors.stack()
    #     observables = observables.stack()
    #     return distr_errors, observables

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def _count_sampled_grid_coordinates(self, samples):
        n_samples = tf.shape(samples)[0]
        zeros = tf.zeros(shape=[self.env.grid_length] * self.env.grid_dimension, dtype=tf.float32)
        updates = tf.ones(shape=[n_samples], dtype=tf.float32)
        counts = tf.tensor_scatter_nd_add(zeros, samples, updates)
        return counts

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
    def _training_step(self, half_batch_size):
        ts1, ba1, fa1 = self._generate_backward_trajectories(half_batch_size)
        ts2, ba2, fa2 = self._generate_forward_trajectories(half_batch_size, training=tf.constant(True))

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
    @tf.function(input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    ])
    def _calculate_ave_loss(log_Z0, log_rewards, action_log_proba_ratios):
        loss = tf.math.square(log_Z0 - log_rewards + action_log_proba_ratios)
        ave_loss = tf.reduce_mean(loss)
        return ave_loss

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
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

    @tf.function(input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.bool)
    ])
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


@tf.function
def _compute_entropy(prob):
    entropy = -tf.reduce_sum(
        tf.where(
            tf.not_equal(prob, 0.0),
            prob*tf.math.log(prob),
            0.0
        )
    )
    return entropy


@tf.function
def _compute_js_dist(prob1, prob2):
    js_div = (
        _compute_entropy(0.5*(prob1 + prob2))
        - 0.5*(
            _compute_entropy(prob1) +
            _compute_entropy(prob2)
        )
    )
    js_dist = tf.math.sqrt(js_div / tf.math.log(2.0))
    return js_dist
