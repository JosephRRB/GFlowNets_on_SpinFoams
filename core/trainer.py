import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from core.environment import SingleVertexSpinFoam, StarModelSpinFoam
from core.runner import Runner
from core.utils import print_and_log

ROOT_DIR = Path(os.path.abspath(__file__ + "/../../"))

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


def train_gfn(
    *,
    model_save_dir,
    logging_file=None,
    spin_j=3.5, 
    sf_model="single_vertex_model",
    main_layer_hidden_nodes=(30, 20), 
    branch1_hidden_nodes=(), 
    branch2_hidden_nodes=(),
    activation="swish",
    exploration_rate=0.5,
    training_fraction_from_back_traj=0.0,
    learning_rate=0.0005,
    batch_size=int(1e3),
    n_iterations=int(1e4),
    evaluation_batch_size=int(1e6),
    generate_samples_every_m_training_samples=int(1e6),
):
    """
    Train a GFlowNet agent to sample grid coordinates with probabilities proportional 
    to a target reward distribution. The SpinFoam model defines the grid environment 
    which the agent interacts with. The grid coordinates then correspond to an ordered 
    set of boundary intertwiners while the square of the SpinFoam amplitudes define the 
    reward for each grid coordinate. This results in grid coordinates with higher rewards
    being sampled more frequently so that the corresponding boundary intertwiners can be
    used for the calculation of observables downstream.
    
    Parameters
    ----------
    
    spin_j: (float)
                - Half-integer boundary spin. All boundary spins are fixed to this value
                - Determines the grid length. All sides of the grid have length: 
                  grid_length = 2*spin_j + 1
                
    sf_model: (str)
                - Name specifying the particular spinfoam model. Currently, can only be
                  "single_vertex_model" or "star_model". Custom SpinFoam classes can also
                  be implemented. 
                - Determines the number of boundary intertwiners which in turn determines the
                  dimensionality of the grid. 
                - E.g. "single_vertex_model" -> 5-dimensional grid: grid_dim = 5, 
                       "star_model" -> 20-dimensional grid: grid_dim = 20
                  
    main_layer_hidden_nodes: (tuple)
                - Tuple of integers defining the number of hidden nodes per layer after the 
                  input layer in the main neural network branch. This main branch splits into
                  two branches corresponding to backward and forward action probabilities. 
                  So, the last layer of the main branch connects to both the first layers of
                  the backward-action and forward-action branches.
                - Input layer contains grid_length*grid_dim nodes because of the one-hot 
                  encoding of each grid coordinate
                - E.g. (128, 64, 32) -> 3 layers after the input layer in the main branch: 
                                        128 nodes for the first one, 64 for the next, and 32
                                        for the last
                                        
    branch1_hidden_nodes: (tuple)
                - Tuple of integers (or empty tuple) defining the number of hidden nodes per 
                  layer in the backward-action branch before the output layer
                - Output layer for this branch contains grid_dim nodes
                - E.g. (16, 8) -> 2 layers before the output layer:
                                  16 for the first one, then 8 for the next
                       () -> there is only the output layer in this branch
                       
    branch2_hidden_nodes: (tuple)
                - Tuple of integers (or empty tuple) defining the number of hidden nodes per 
                  layer in the forward-action branch before the output layer
                - Output layer for this branch contains grid_dim+1 nodes
                - E.g. (16, 8) -> 2 layers before the output layer:
                                  16 for the first one, then 8 for the next
                       () -> there is only the output layer in this branch
                       
    activation: (str)
                - Name of the nonlinear functions used for each layer other than the output layers
                - E.g.: elu, exponential, gelu, hard_sigmoid, linear, relu, selu, sigmoid, softmax, 
                        softplus, softsign, swish, tanh
                        
    exploration_rate: (float)
                - Number from 0 to 1 representing the amount of random noise added to the forward 
                  actions. More noise encourages the agent to explore the grid environment further
                - Given by: New_action_proabilities = (
                                exploration_rate*noise_action_probabilities + 
                                (1 - exploration_rate)*current_action_probabilities
                            )
                - I.e.: exploration_rate = 0 -> no noise added
                        exploration_rate = 1 -> actions are just from random noise
                        
    training_fraction_from_back_traj: (float)
                - Number from 0 to 1 representing the fraction of the training set which is 
                  generated from backward trajectories. The rest is generated from forward trajectories.
                - I.e.: n_batch_backwards = training_fraction_from_back_traj*batch_size
                        n_batch_forwards = (1 - training_fraction_from_back_traj)*batch_size
                        
    learning_rate: (float)
                - Represents the amount of the gradient used by the optimizer to update the neural 
                  network parameters per training iteration 
                  
    batch_size: (int)
                - Number of trajectories generated per training iteration. These trajectories form
                  the training set that the agent learns from. 
                  
    n_iterations: (int)
                - Number of training iterations for the agent
                
    evaluation_batch_size: (int)
                - Number of grid coordinates that the agent generates and stores for later evaluation
                
    generate_samples_every_m_training_samples: (int)
                - Number of trajectories generated during training before the agent samples new 
                  grid coordinates and stores them
    """
    if sf_model == "single_vertex_model":
        spinfoam_model = SingleVertexSpinFoam(spin_j=spin_j)
    elif sf_model == "star_model":
        spinfoam_model = StarModelSpinFoam(spin_j=spin_j)
    else:
        raise ValueError(
            "Spinfoam model not yet implemented. "
            "Custom Spinfoam class can be made."
        )
    
    runner = Runner(
        spinfoam_model=spinfoam_model,
        main_layer_hidden_nodes=main_layer_hidden_nodes,
        branch1_hidden_nodes=branch1_hidden_nodes,
        branch2_hidden_nodes=branch2_hidden_nodes,
        activation=activation,
        exploration_rate=exploration_rate,
        training_fraction_from_back_traj=training_fraction_from_back_traj,
        learning_rate=learning_rate
    )
    
    ave_losses = runner.train_agent(
        batch_size, n_iterations, 
        evaluation_batch_size, generate_samples_every_m_training_samples,
        model_save_dir, logging_file
    )
    
    return ave_losses


def train_models(
    modelparams
) :
    """
    Trains the agent and saves the model parameters and training data to a directory.
    
    Parameters
    ----------
    
    modelparams: (dict)
                - Dictionary containing the model parameters. These are the parameters 
                  for the SpinFoam model and the neural network architecture. 
                - E.g.: modelparams = {
                            "spin_j": 1.5,
                            "sf_model": "single_vertex_model",
                            "main_layer_hidden_nodes": (128, 64, 32),
                            "branch1_hidden_nodes": (16, 8),
                            "branch2_hidden_nodes": (16, 8),
                            "activation": "elu",
                            "exploration_rate": 0.1,
                            "training_fraction_from_back_traj": 0.5,
                            "learning_rate": 0.001,
                            "batch_size": 100,
                            "n_iterations": 1000,
                            "evaluation_batch_size": 1000,
                            "generate_samples_every_m_training_samples": 10,
                        }
                        
    model_directory: (str)
                - Path to the directory where the model parameters and training data will be saved
    """
    # Create the directory for the models
    model_base_directory = Path(ROOT_DIR, "models", modelparams.sf_model[0])
    os.makedirs(model_base_directory, exist_ok=True)
    
    num_models = len(modelparams)
    for idx, params in enumerate(modelparams):
        # Load current models in the directory, if any.
        # The next model will be saved with the next available number
        model_dir = model_base_directory / Path(f"model_{len(os.listdir(model_base_directory))}")
        os.makedirs(model_dir, exist_ok=True)
        
        logging_file = model_dir / Path("training_log.txt")
        logging_file.touch()
    
        model_results_file = model_dir / Path("results.npy")
        model_results_file.touch()
        
        # Save the model parameters to file
        with open(model_dir / Path("modelparams.json"), "w") as f:
            json.dump(params, f)
        
        # Train the model
        print_and_log(f"Testing model: {params['sf_model']}\n", logging_file)
        print_and_log(f"Starting training for parameter set {idx} of {num_models}\n", logging_file)
        
        training_start = datetime.now()
        avg_losses = train_gfn(model_save_dir=model_dir, logging_file=logging_file, **params)
        training_time = (datetime.now() - training_start).total_seconds()
        
        print_and_log(f"\nFinished training, elapsed time: {training_time / 60:.2f} minutes\n", logging_file)

        np.save(model_dir / Path("results.npy"), np.array([training_time, avg_losses]))