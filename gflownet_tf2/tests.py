from gfn import GFNAgent

def test():
    agent = GFNAgent(epochs=200)
    agent.sample(10)
    agent.train()
    l1_error_after = agent.compare_env_to_model_policy()
    agent

