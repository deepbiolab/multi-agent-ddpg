import numpy as np
from .env_wrapper import SubprocVecEnv
from .environment import MultiAgentEnv
import src.multiagent.scenarios as scenarios

def make_parallel_env(n_rollout_threads, seed=1):
    def get_env_fn(rank):
        def init_env():
            env = make_env("simple_adversary")
            # Set both numpy and environment seeds
            np.random.seed(seed + rank * 1000)
            env.seed(seed + rank * 1000)
            return env
        return init_env
    return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def make_env(scenario_name, benchmark=False):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env