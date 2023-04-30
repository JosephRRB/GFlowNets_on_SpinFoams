# import os

import tensorflow as tf
import numpy as np

from core.environment import HypergridEnvironment, _calculate_dihedral_angles
from core.agent import Agent

# ROOT_DIR = os.path.abspath(__file__ + "/../../")
# WORKING_DIR = f"{ROOT_DIR}/working_directory"
# os.makedirs(WORKING_DIR, exist_ok=True)

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
    ])
    def train_agent(
            self, batch_size, n_iterations, check_loss_every_n_iterations
    ):
        ave_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        training_samples = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        for i in tf.range(n_iterations):
            ave_loss, terminal_states = self._training_step(batch_size)
            # np.savetxt(
            #     f"{WORKING_DIR}/Epoch #{i+1} Training Samples.csv",
            #     terminal_states.numpy(), delimiter=","
            # )

            ave_losses = ave_losses.write(i, ave_loss)
            training_samples = training_samples.write(i, terminal_states)

            if tf.math.equal(tf.math.floormod(i+1, check_loss_every_n_iterations), 0):
                tf.print("Nth iteration:",  i+1, "Average Loss:", ave_loss)

        ave_losses = ave_losses.stack()
        training_samples = training_samples.stack()
        return ave_losses, training_samples

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

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def _count_sampled_grid_coordinates(self, samples):
        n_samples = tf.shape(samples)[0]
        zeros = tf.zeros(shape=[self.env.grid_length] * self.env.grid_dimension, dtype=tf.float32)
        updates = tf.ones(shape=[n_samples], dtype=tf.float32)
        counts = tf.tensor_scatter_nd_add(zeros, samples, updates)
        return counts

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
    def _training_step(self, batch_size):
        trajectories, backward_actions, forward_actions = \
            self._generate_forward_trajectories(
                batch_size, training=tf.constant(True)
            )
        terminal_states = trajectories[-1]
        rewards = self.env.get_rewards(terminal_states)
        with tf.GradientTape() as tape:
            action_log_proba_ratios = \
                self.agent.calculate_action_log_probability_ratio(
                    trajectories, backward_actions, forward_actions
                )
            log_Z0 = self.agent.log_Z0
            log_rewards = tf.math.log(rewards)
            ave_loss = self._calculate_ave_loss(
                log_Z0, log_rewards, action_log_proba_ratios
            )

        grads = tape.gradient(
            ave_loss, self.agent.policy.trainable_weights + [self.agent.log_Z0]
        )
        self.optimizer.apply_gradients(
            zip(grads, self.agent.policy.trainable_weights + [self.agent.log_Z0])
        )
        return ave_loss, terminal_states


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
