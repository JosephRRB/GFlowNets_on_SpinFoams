import datetime
import itertools
import pickle
from dataclasses import dataclass

import tensorflow as tf

from core.trainer import train_gfn


@dataclass
class ModelParams:
    spin_j: float 
    sf_model: str
    main_layer_hidden_nodes: tuple
    branch1_hidden_nodes: tuple
    branch2_hidden_nodes: tuple
    activation: str
    exploration_rate: float
    training_fraction_from_back_traj: float
    learning_rate: float
    batch_size: int
    n_iterations: int
    evaluation_batch_size: int
    generate_samples_every_m_training_samples: int
    _params_list: list = None
    _idx: int = 0
    
    def __post_init__(self, *args, **kwargs):
        self._params_list = list(itertools.product(*self.params()))
    
    def fields(self):
        return list(filter(
            lambda x: x.startswith("_") == False,
            self.__dataclass_fields__.keys()
        ))

    def params(self):
        return (getattr(self, field, ()) for field in self.fields())
    
    def __len__(self):
        return len(self._params_list)
    
    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration

        p = {
            key: value
            for key, value in zip(self.fields(), self._params_list[self._idx])
        }
        self._idx += 1
        return p

single_model = ModelParams(
        sf_model = ["single_vertex_model"], # Input layer: 5 * (2 * spin  + 1), Output layer: forward = 5 + 1, backward = 5
        spin_j = [3.5, 6.5],
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
        spin_j = [3.5, 6.5],
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
total_number_of_models = sum(map(len, models))
tf.print(f"Total number of models:", total_number_of_models)
tf.print(f"Expected time to complete:", total_number_of_models * 5 / 60, "hours")

models_avg_losses = {
    model.sf_model[0]: []
    for model in models
}

start = datetime.datetime.now()
tf.print("Starting testing:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "\n")

for model in models:
    num_models = len(model)
    tf.print("Testing model:", model.sf_model[0], "\n")
    with open(f"results/{model.sf_model[0]}-params.pkl", "wb") as f:
        f.write("training_time, params, avg_losses\n")
    for i, params in enumerate(model):
        tf.print(f"Starting training for parameter set {i} of {num_models}")
        training_start = datetime.datetime.now()
        avg_losses = train_gfn(**params, include_datetime_in_directory_name=False)
        training_time = (datetime.datetime.now() - training_start).total_seconds()
        tf.print("Finished training, elapsed time:", training_time / 60, "minutes")
        models_avg_losses[model.sf_model[0]].append((training_time, params, avg_losses))

with open(f"results/{model.sf_model[0]}-params.pkl", "ab") as f:
    pickle.dump(f, training_time, params, avg_losses)

tf.print("Finished testing:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
tf.print("Total time taken: ", (datetime.datetime.now() - start).total_seconds() / 60, "minutes")
tf.print("Saving results")
