import time
import os
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Import our standard config and the Team enum.
from pyquaticus.config import config_dict_std
from pyquaticus.envs.pyquaticus import Team
from pyquaticus.envs.coach_env import CoachEnv

# Import our custom wrapper.
from my_pettingzoo_env import MyPettingZooEnv

def env_creator_coach(env_config):
    """
    Creates the underlying PyQuaticus environment, wraps it with our custom wrapper
    (MyPettingZooEnv) to add missing attributes, builds the agent_team_map, and then
    instantiates and returns a CoachEnv.
    """
    # Import the module that provides the original PyQuaticusEnv.
    from pyquaticus import pyquaticus_v0

    # Copy the standard config and update parameters as needed.
    config_dict = config_dict_std.copy()
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True

    # For a 3v3 game, team_size=3 creates 6 agents.
    underlying_env = pyquaticus_v0.PyQuaticusEnv(
        config_dict=config_dict,
        render_mode=None,
        reward_config=None,
        team_size=3
    )

    # Wrap the underlying environment so that it supplies the missing attributes.
    wrapped_env = MyPettingZooEnv(underlying_env)

    # Build an agent_team_map from the underlying environment's players.
    # (This assumes that your original PyQuaticusEnv has a 'players' dictionary.)
    agent_team_map = {agent_id: player.team for agent_id, player in underlying_env.players.items()}

    # Create and return the CoachEnv.
    coach_env = CoachEnv(
        team=env_config["team"],
        env_creator=lambda: wrapped_env,  # You can return the wrapped env here.
        agent_team_map=agent_team_map
    )
    return coach_env

# Register the custom environment with RLlib.
register_env("coach_env", env_creator_coach)

def train_coach():
    # Set the configuration for the coach environment.
    env_config = {"team": Team.RED_TEAM}
    config = PPOConfig().environment(env="coach_env", env_config=env_config).framework("torch")
    algo = config.build_algo()

    # Create a directory for saving checkpoints.
    checkpoint_dir = os.path.abspath("./ray_test")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop.
    for i in range(10):
        print(f"Training Iteration: {i}")
        start_time = time.time()
        algo.train()
        end_time = time.time()
        print(f"Iteration {i} complete. Time Taken: {end_time - start_time:.2f} seconds")

        # Save a checkpoint every 5 iterations.
        if i % 5 == 0:
            cp_path = os.path.join(checkpoint_dir, f"iter_{i}")
            os.makedirs(cp_path, exist_ok=True)
            saved_checkpoint = algo.save(cp_path)
            print(f"Checkpoint saved at: {saved_checkpoint}")

if __name__ == '__main__':
    train_coach()
