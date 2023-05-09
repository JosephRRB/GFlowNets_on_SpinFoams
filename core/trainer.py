from datetime import datetime

from core.runner import Runner
from core.environment import SingleVertexSpinFoam, StarModelSpinFoam

def train_gfn(
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
    include_datetime_in_directory_name=True,
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
    
    directory_for_generated_samples = (
        f"generated_samples_during_training/{sf_model}/j={spin_j}/"
    )
    if include_datetime_in_directory_name:
        training_run_datetime = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        directory_for_generated_samples += f"Training run on {training_run_datetime}/"
    
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
        directory_for_generated_samples
    )
    
    return ave_losses
    