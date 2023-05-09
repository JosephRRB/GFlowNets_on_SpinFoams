from core.trainer import train_gfn
from core.runner import Runner
from core.environment import SingleVertexSpinFoam
from datetime import datetime
import matplotlib.pyplot as plt

ave_losses = dict()

# Single Vertex
ave_losses = train_gfn(
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
)
plt.plot(ave_losses)

# Star Model
ave_losses = train_gfn(
    spin_j=3.5, 
    sf_model="star_model",
    main_layer_hidden_nodes=(128, 64, 64), 
    branch1_hidden_nodes=(), 
    branch2_hidden_nodes=(),
    activation="swish",
    exploration_rate=0.5,
    training_fraction_from_back_traj=0.5,
    learning_rate=0.0005,
    batch_size=int(1e3),
    n_iterations=int(1e4),
    evaluation_batch_size=int(1e6),
    generate_samples_every_m_training_samples=int(1e6),
)
plt.plot(ave_losses)