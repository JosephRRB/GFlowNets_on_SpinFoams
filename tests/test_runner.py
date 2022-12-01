import tensorflow as tf

from core.runner import Runner
from core.environment import HypergridEnvironment
from core.agent import Agent

def test():
    grid_dim = 5
    grid_length = 8
    env = HypergridEnvironment(grid_dimension=grid_dim, grid_length=grid_length)
    agent = Agent(env_grid_dim=grid_dim, env_grid_length=grid_length)

    runner = Runner(agent=agent, environment=env)
    trajectories, backward_actions, forward_actions = runner.generate_backward_trajectories(10)