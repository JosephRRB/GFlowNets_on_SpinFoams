from core.trainer import train_gfn
from core.runner import Runner
from core.environment import SingleVertexSpinFoam
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Parameters to test
from dataclasses import dataclass
import itertools

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
    
    def fields(self):
        return list(self.__dataclass_fields__.keys())
    
    def params(self):
        return (getattr(self, field, ()) for field in self.fields())
    
    def __len__(self):
        return len(list(itertools.product(*self.params())))
    
    def __iter__(self):
        return self

    def __next__(self):        
        for param in itertools.product(*self.params()):
            return {
                key: value
                for key, value in zip(self.fields(), param)
            }
            
        raise StopIteration 

single_model = ModelParams(
        sf_model = ["single_vertex_model"], # Input layer: 5 * (2 * spin  + 1), Output layer: forward = 5 + 1, backward = 5
        spin_j = [3, 4, 5, 6],
        main_layer_hidden_nodes = [(64, 32, 16, 8), (64, 32, 32), (64, 16), (64, 64, 16, 16)],
        branch1_hidden_nodes = [()],
        branch2_hidden_nodes = [()],
        activation = ["swish", "tanh", "relu"],
        exploration_rate = [0.5],
        training_fraction_from_back_traj = [0.0],
        learning_rate = [0.0005],
        batch_size = [1e3],
        n_iterations = [1e4],
        evaluation_batch_size = [1e6],
        generate_samples_every_m_training_samples = [1e6],
)

star_model = ModelParams(
        sf_model = ["star_model"],
        spin_j = [3.5, 6.5],
        main_layer_hidden_nodes = [(256, 128, 64, 32), (256, 64, 64, 32), (256, 192, 64, 32)],
        branch1_hidden_nodes = [()],
        branch2_hidden_nodes = [()],
        activation = ["swish", "tanh", "relu"],
        exploration_rate = [0.5],
        training_fraction_from_back_traj = [1.0],
        learning_rate = [0.0005],
        batch_size = [1e3],
        n_iterations = [1e4],
        evaluation_batch_size = [1e6],
        generate_samples_every_m_training_samples = [1e6],
)

models = [single_model, star_model]

total_number_of_models = sum(map(len, models))
print(f"Total number of models: {total_number_of_models} to run.")

ave_losses = {
    "single_vertex_model": [],
    "star_model": [],
}

for model in models:
    for params in model:
        ave_losses = train_gfn(**params)
        ave_losses[model.sf_model].append((params, ave_losses))