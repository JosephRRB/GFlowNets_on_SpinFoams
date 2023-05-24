import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from core.agent import Agent
from core.environment import BaseSpinFoam, SpinFoamEnvironment
from core.policy_network import PolicyNetwork
from core.utils import print_and_log

ROOT_DIR = Path(os.path.abspath(__file__ + "/../../"))


class Runner:
    def __init__(
        self,
        spinfoam_model: BaseSpinFoam,
        main_layer_hidden_nodes=(30, 20),
        branch1_hidden_nodes=(10,),
        branch2_hidden_nodes=(10,),
        activation="swish",
        exploration_rate=0.5,
        training_fraction_from_back_traj=0.0,
        learning_rate=0.0005,
    ):
        self.env = SpinFoamEnvironment(spinfoam_model=spinfoam_model)
        self.agent = Agent(
            self.env.grid_dimension,
            self.env.grid_length,
            main_layer_hidden_nodes=main_layer_hidden_nodes,
            branch1_hidden_nodes=branch1_hidden_nodes,
            branch2_hidden_nodes=branch2_hidden_nodes,
            activation=activation,
            exploration_rate=exploration_rate,
        )
        self.agent.set_initial_estimate_for_log_z0(
            self.env.spinfoam_model.single_vertex_amplitudes,
            self.env.spinfoam_model.n_vertices,
            frac=0.99,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.training_fraction_from_back_traj = training_fraction_from_back_traj

    def train_agent(
        self,
        training_batch_size,
        n_iterations,
        evaluation_batch_size,
        generate_samples_every_m_training_samples,
        base_data_folder,
        logging_file=None
    ):
        generated_samples_dir = Path(base_data_folder, "generated_samples")
        os.makedirs(generated_samples_dir, exist_ok=True)

        ave_losses = tf.TensorArray(dtype=tf.float64, size=0, dynamic_size=True)
        if generate_samples_every_m_training_samples % training_batch_size != 0:
            raise ValueError(
                f"Evaluate only in multiples of "
                f"training_batch_size={training_batch_size}"
            )

        n_batch_backwards = int(
            self.training_fraction_from_back_traj * training_batch_size
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
            ave_losses = ave_losses.write(i, ave_loss)

            trained_on_k_samples = (i + 1) * training_batch_size
            if trained_on_k_samples % generate_samples_every_m_training_samples == 0:
                print_and_log(
                    f"{i + 1}th iteration trained on {trained_on_k_samples} samples with average loss {ave_loss}",
                    logging_file
                )
                samples = self.generate_samples_from_agent(evaluation_batch_size)
                
                samples_file = Path(
                    f"{generated_samples_dir}/"
                    f"epoch_{i + 1}"
                    f"_after_learn_from_{trained_on_k_samples}"
                    "_train_samples.csv"
                )
                samples_file.touch()

                header = ",".join(
                    [f"intertwiner {i+1}" for i in reversed(range(self.env.spinfoam_model.n_boundary_intertwiners))]
                )

                np.savetxt(
                    samples_file,
                    samples.numpy(),
                    delimiter=",",
                    header=header,
                    fmt="%i",
                    comments="",
                )

        ave_losses = ave_losses.stack()

        avg_loss_file = Path(base_data_folder, "average_losses.csv")
        avg_loss_file.touch()
        np.savetxt(
            avg_loss_file,
            ave_losses.numpy(),
            header="average_loss",
            delimiter=",",
            fmt="%f",
            comments="",
        )

        self.save_agent_policy(base_data_folder)
        return ave_losses

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)])
    def generate_samples_from_agent(self, batch_size):
        """
        Using the current agent, generate a batch of samples. These samples are
        the terminal states of forward trajectories starting from the grid
        origin: [0] * grid_dim. For each iteration, pass the batch of
        intermediate grid coordinates to the agent which will return actions
        representing which grid coordinate to increment forward (plus an
        additional action to terminate the sequence). With these actions, the
        environment will update the current state. This process is repeated
        until all the trajectories have been terminated.

        Similar to _generate_forward_trajectories() except that only the
        terminal states are returned.

        Parameters:
        -----------
        batch_size:
            Number of samples to generate

        Return:
        -------
        current_position:
            The batch of generated samples. These are the terminal states of the
            forward trajectories generated by the agent.
        """
        current_position = self.env.reset_for_forward_sampling(batch_size)
        is_still_sampling = tf.ones(shape=(batch_size, 1), dtype=tf.int32)
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

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.int32),
        ]
    )
    def _training_step(self, n_batch_forwards, n_batch_backwards):
        """
        Using the current agent, generate a batch of trajectories, calculate the
        loss function on those trajectories, and estimate the gradients. Then
        update the trainable parameters using those gradients.

        Parameters:
        -----------
        n_batch_forwards:
            Number of forward trajectories generated by the current model

        n_batch_backwards:
            Number backward trajectories generated by the current model

        Return:
        -------
        ave_loss:
            The loss function averaged over all the generated trajectories
        """
        terminal_states = tf.constant(
            [], dtype=tf.int32, shape=(0, self.env.grid_dimension)
        )
        action_log_proba_ratios = tf.constant([], dtype=tf.float64, shape=(0, 1))
        forward_trajectories = (tf.constant([], dtype=tf.int32),) * 3
        backward_trajectories = (tf.constant([], dtype=tf.int32),) * 3
        if tf.math.not_equal(n_batch_forwards, 0):
            forward_trajectories = self._generate_forward_trajectories(
                n_batch_forwards, training=tf.constant(True)
            )
            terminal_states = tf.concat(
                [terminal_states, forward_trajectories[0][-1]], axis=0
            )

        if tf.math.not_equal(n_batch_backwards, 0):
            backward_trajectories = self._generate_backward_trajectories(
                n_batch_backwards
            )
            terminal_states = tf.concat(
                [terminal_states, backward_trajectories[0][0]], axis=0
            )

        rewards = self.env.get_rewards(terminal_states)
        with tf.GradientTape() as tape:
            if tf.math.not_equal(n_batch_forwards, 0):
                action_log_proba_ratios_ft = (
                    self.agent.calculate_action_log_probability_ratio(
                        *forward_trajectories
                    )
                )
                action_log_proba_ratios = tf.concat(
                    [action_log_proba_ratios, action_log_proba_ratios_ft], axis=0
                )

            if tf.math.not_equal(n_batch_backwards, 0):
                action_log_proba_ratios_bt = (
                    self.agent.calculate_action_log_probability_ratio(
                        *backward_trajectories
                    )
                )
                action_log_proba_ratios = tf.concat(
                    [action_log_proba_ratios, action_log_proba_ratios_bt], axis=0
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
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
        ]
    )
    def _calculate_ave_loss(log_Z0, log_rewards, action_log_proba_ratios):
        """
        Calculate the Trajectory Balance loss averaged over all the trajectories
        generated during the training step.
        """
        loss = tf.math.square(log_Z0 - log_rewards + action_log_proba_ratios)
        ave_loss = tf.reduce_mean(loss)
        return ave_loss

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.int32),
            tf.TensorSpec(shape=None, dtype=tf.bool),
        ]
    )
    def _generate_forward_trajectories(self, batch_size, training):
        """
        Generate a batch of forward trajectories starting from the grid origin:
        [0] * grid_dim. For each iteration, pass the batch of intermediate grid
        coordinates to the agent which will return actions representing which
        grid coordinate to increment forward (plus an additional action to
        terminate the sequence). With these actions, the environment will update
        the current state. This process is repeated until all the trajectories
        have been terminated.

        Similar to generate_samples_from_agent() except that all the
        intermediate states and actions are returned.

        Parameters:
        -----------
        batch_size:
            Number of trajectories to generate

        training:
            Boolean flag for the agent. If training=True, agent will add noise
            to the action probabilities it generates. This will encourage
            exploration of the environment grid.

        Return:
        -------
        trajectories:
            - 3D Tensor of shape (max_traj_len, batch_size, grid_dim)
            - Batch of trajectories. Shorter trajectories are padded with their
              terminal states until the last trajectory has terminated. That is,
              trajectories that terminated earlier will no longer increment
              their states.
            - The last element of the sequence trajectories[-1] is considered to
              be the batch of terminal states.

        forward_actions:
            - 3D Tensor of shape (max_traj_len, batch_size, grid_dim + 1)
            - Sequence of one-hot encoded actions corresponding to which
              coordinate to increment (and an additional terminate action)
            - The actions forward_actions[i] induce the forward state
              transition: trajectories[i] -> trajectories[i+1]. Once the
              terminate action has been selected, the next actions are all zero.

        backward_actions:
            - 3D Tensor of shape (max_traj_len, batch_size, grid_dim)
            - Sequence of one-hot encoded actions corresponding to which
              coordinate was incremented to reach the current state.
            - The actions backward_actions[i] refer to the actions used in the
              state transition: trajectories[i-1] -> trajectories[i]. Since all
              trajectories start from the grid origin, backward_actions[0] is
              defined to be all zero.
        """
        current_position = self.env.reset_for_forward_sampling(batch_size)
        is_still_sampling = tf.ones(shape=(batch_size, 1), dtype=tf.int32)

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
        """
        Generate a batch of backward trajectories starting from grid coordinates
        chosen uniformly. For each iteration, pass the batch of intermediate
        grid coordinates to the agent which will return actions representing
        which grid coordinate to decrement. With these actions, the environment
        will update the current state. Repeat the process until all trajectories
        have reached the grid origin: [0] * grid_dim.

        Parameters:
        -----------
        batch_size:
            Number of trajectories to generate

        Return:
        -------
        trajectories:
            - 3D Tensor of shape (max_traj_len, batch_size, grid_dim)
            - Batch of trajectories. Shorter trajectories are padded with zeros
              until the last trajectory has reached the grid origin. That is,
              trajectories that reach the grid origin earlier will no longer
              decrement their states.
            - The first element of the sequence trajectories[0] is considered
              to be the batch of terminal states.

        backward_actions:
            - 3D Tensor of shape (max_traj_len, batch_size, grid_dim)
            - Sequence of one-hot encoded actions corresponding to which
              coordinate to decrement.
            - The actions backward_actions[i] induce the backward state
              transition: trajectories[i] -> trajectories[i+1]. Once the grid
              origin is reached, the next actions are all zero. Since
              trajectories[-1] are all the grid origin, backward_actions[-1] is
              defined to be all zero.

        forward_actions:
            - 3D Tensor of shape (max_traj_len, batch_size, grid_dim + 1)
            - Sequence of one-hot encoded actions corresponding to which
              coordinate was decremented to reach the current state.
            - The actions forward_actions[i] refer to the actions used in the
              state transition: trajectories[i-1] -> trajectories[i]. Since all
              trajectories start from terminal state, forward_actions[0] is
              defined to be all terminate actions.
        """
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

            trajectories = trajectories.write(i + 1, current_position)
            backward_actions = backward_actions.write(i, action)
            forward_actions = forward_actions.write(i + 1, action)

            i += 1
            at_least_one_ongoing = tf.math.reduce_any(
                tf.math.not_equal(current_position, 0)
            )
        backward_actions = backward_actions.write(i, no_coord_action)

        trajectory_max_len = i + 1
        stop_actions = tf.scatter_nd(
            indices=[[0]],
            updates=tf.ones(shape=(1, batch_size, 1), dtype=tf.int32),
            shape=(trajectory_max_len, batch_size, 1),
        )
        trajectories = trajectories.stack()
        backward_actions = backward_actions.stack()
        forward_actions = tf.concat([forward_actions.stack(), stop_actions], axis=2)
        return trajectories, backward_actions, forward_actions

    def save_agent_policy(self, path):
        self.agent.policy.save(Path(f"{path}/trained_agent_policy"))

    def load_agent_policy(self, path):
        return tf.keras.models.load_model(
            Path(f"{path}/trained_agent_policy"),
            custom_objects={"PolicyNetwork": PolicyNetwork}
        )