import datetime
import os

import tensorflow as tf

from core.trainer import ModelParams, train_models

single_model = ModelParams(
        sf_model = ["single_vertex_model"], # Input layer: 5 * (2 * spin  + 1), Output layer: forward = 5 + 1, backward = 5
        spin_j = [3.5, 6.0],
        main_layer_hidden_nodes = [(64, 32, 16, 8), (64, 32, 32), (64, 16), (64, 64, 16, 16)],
        branch1_hidden_nodes = [()],
        branch2_hidden_nodes = [()],
        activation = ["swish", "tanh", "relu"],
        exploration_rate = [0.5],
        training_fraction_from_back_traj = [0.0],
        learning_rate = [0.0005],
        batch_size = [int(1e3)],
        n_iterations = [int(1e4)],
        evaluation_batch_size = [int(1e6)],
        generate_samples_every_m_training_samples = [int(1e6)],
)

star_model = ModelParams(
        sf_model = ["star_model"], # Input layer: 20 * (2 * spin  + 1), Output layer: forward = 20 + 1, backward = 20
        spin_j = [3.5, 6.0],
        main_layer_hidden_nodes = [(256, 128, 64, 32), (256, 64, 64, 32), (256, 32), (256, 192, 64, 32)],
        branch1_hidden_nodes = [()],
        branch2_hidden_nodes = [()],
        activation = ["swish", "tanh", "relu"],
        exploration_rate = [0.5],
        training_fraction_from_back_traj = [1.0],
        learning_rate = [0.0005],
        batch_size = [int(1e3)],
        n_iterations = [int(1e4)],
        evaluation_batch_size = [int(1e6)],
        generate_samples_every_m_training_samples = [int(1e6)],
)

models = [single_model, star_model]

total_number_of_models = sum(map(len, models))
tf.print(f"\n\nTotal number of models:", total_number_of_models)
tf.print(f"Expected time to complete (5mins per model):", total_number_of_models * 5 / 60, "hours\n")

start = datetime.datetime.now()
tf.print("Starting testing:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "\n")

for model in models:
    train_models(model)

tf.print("Finished testing:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
tf.print(f"Total time taken: {(datetime.datetime.now() - start).total_seconds() / 60:.2f} minutes\n")
