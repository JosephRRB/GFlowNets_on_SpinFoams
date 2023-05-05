import os

import tensorflow as tf
import numpy as np

from core.environment import SpinFoamEnvironment, BaseSpinFoam
from core.agent import Agent

ROOT_DIR = os.path.abspath(__file__ + "/../../")


class Runner:
    def __init__(self,
                 spinfoam_model: BaseSpinFoam,
                 main_layer_hidden_nodes=(30, 20),
                 branch1_hidden_nodes=(10, ),
                 branch2_hidden_nodes=(10, ),
                 activation="swish",
                 exploration_rate=0.5,
                 learning_rate=0.0005
                 ):
        self.env = SpinFoamEnvironment(spinfoam_model=spinfoam_model)

        self.agent = Agent(
            self.env.grid_dimension, self.env.grid_length,
            main_layer_hidden_nodes=main_layer_hidden_nodes,
            branch1_hidden_nodes=branch1_hidden_nodes,
            branch2_hidden_nodes=branch2_hidden_nodes,
            activation=activation,
            exploration_rate=exploration_rate
        )
        self.agent.set_initial_estimate_for_log_z0(
            self.env.spinfoam_model.single_vertex_amplitudes,
            self.env.spinfoam_model.n_vertices,
            frac=0.99
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_agent(
            self, training_batch_size, n_iterations,
            fraction_of_training_from_back_trajectories,
            evaluation_batch_size, generate_samples_every_m_training_samples,
            directory_for_generated_samples
    ):
        filepath = f"{ROOT_DIR}/{directory_for_generated_samples}"
        os.makedirs(filepath, exist_ok=True)

        ave_losses = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
        if generate_samples_every_m_training_samples % training_batch_size != 0:
            raise ValueError(
                f"Evaluate only in multiples of "
                f"training_batch_size={training_batch_size}"
            )

        n_batch_backwards = int(
            fraction_of_training_from_back_trajectories * training_batch_size
        )
        n_batch_forwards = training_batch_size - n_batch_backwards

        if n_batch_backwards < 0 or n_batch_forwards < 0:
            raise ValueError(
                f"The fraction of backward trajectories in the training batch "
                f"should be between 0 and 1."
            )

        n_batch_backwards = tf.constant(n_batch_backwards)
        n_batch_forwards = tf.constant(n_batch_forwards)
        evaluation_batch_size = tf.constant(evaluation_batch_size)
        for i in tf.range(n_iterations):
            ave_loss = self._training_step(n_batch_forwards, n_batch_backwards)
            if i == 0:
                tf.print("Nth iteration:",  i+1, "Average Loss:", ave_loss)
            ave_losses = ave_losses.write(i, ave_loss)

            trained_on_k_samples = (i + 1) * training_batch_size
            if trained_on_k_samples % generate_samples_every_m_training_samples == 0:
                tf.print(
                    "Nth iteration:",  i+1,
                    "Trained on K samples:", trained_on_k_samples,
                    "Average Loss:", ave_loss
                )
                samples = self.generate_samples_from_agent(evaluation_batch_size)
                filename = (
                    f"{filepath}/"
                    f"Generated samples for epoch #{i + 1} "
                    f"after learning from {trained_on_k_samples} "
                    "training samples.csv"
                )
                np.savetxt(
                    filename, samples.numpy(), delimiter=",", fmt='%i',
                    # header="i1,i2,i3,i4,i5", comments=""
                )

        ave_losses = ave_losses.stack()
        return ave_losses

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

    @tf.function(input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.int32)
    ])
    def _training_step(self, n_batch_forwards, n_batch_backwards):
        terminal_states = tf.constant(
            [], dtype=tf.int32, shape=(0, self.env.grid_dimension)
        )
        action_log_proba_ratios = tf.constant(
            [], dtype=tf.float64, shape=(0, 1)
        )
        forward_trajectories = (tf.constant([], dtype=tf.int32), ) * 3
        backward_trajectories = (tf.constant([], dtype=tf.int32), ) * 3
        if tf.math.not_equal(n_batch_forwards, 0):
            forward_trajectories = self._generate_forward_trajectories(
                    n_batch_forwards, training=tf.constant(True)
                )
            terminal_states = tf.concat([
                terminal_states, forward_trajectories[0][-1]
            ], axis=0)

        if tf.math.not_equal(n_batch_backwards, 0):
            backward_trajectories = self._generate_backward_trajectories(
                n_batch_backwards
            )
            terminal_states = tf.concat([
                terminal_states, backward_trajectories[0][0]
            ], axis=0)

        rewards = self.env.get_rewards(terminal_states)
        # logr = tf.math.log(rewards)
        # ave_logr = tf.reduce_mean(logr)
        # min_logr = tf.reduce_min(logr)
        # max_logr = tf.reduce_max(logr)
        # tf.print("ave: ", ave_logr, "\n min: ", min_logr, "\n max: ", max_logr)
        with tf.GradientTape() as tape:
            if tf.math.not_equal(n_batch_forwards, 0):
                action_log_proba_ratios_ft = \
                    self.agent.calculate_action_log_probability_ratio(
                        *forward_trajectories
                    )
                action_log_proba_ratios = tf.concat([
                    action_log_proba_ratios, action_log_proba_ratios_ft
                ], axis=0)

            if tf.math.not_equal(n_batch_backwards, 0):
                action_log_proba_ratios_bt = \
                    self.agent.calculate_action_log_probability_ratio(
                        *backward_trajectories
                    )
                action_log_proba_ratios = tf.concat([
                    action_log_proba_ratios, action_log_proba_ratios_bt
                ], axis=0)

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
        return ave_loss


    @staticmethod
    @tf.function(input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
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