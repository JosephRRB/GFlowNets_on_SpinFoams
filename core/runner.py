import os

import tensorflow as tf
import numpy as np

from core.environment import SpinFoamEnvironment
from core.agent import Agent

ROOT_DIR = os.path.abspath(__file__ + "/../../")

class Runner:
    def __init__(self,
                 spin_j=3.5,
                 main_layer_hidden_nodes=(30, 20),
                 branch1_hidden_nodes=(10, ),
                 branch2_hidden_nodes=(10, ),
                 activation="swish",
                 exploration_rate=0.5,
                 learning_rate=0.0005
                 ):
        self.env = SpinFoamEnvironment(spin_j=spin_j)

        self.agent = Agent(
            self.env.grid_dimension, self.env.grid_length,
            main_layer_hidden_nodes=main_layer_hidden_nodes,
            branch1_hidden_nodes=branch1_hidden_nodes,
            branch2_hidden_nodes=branch2_hidden_nodes,
            activation=activation,
            exploration_rate=exploration_rate
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_agent(
            self, training_batch_size, n_iterations,
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

        for i in tf.range(n_iterations):
            ave_loss = self._training_step(training_batch_size)
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
