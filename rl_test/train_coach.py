import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pyquaticus.config import config_dict_std
from pyquaticus.envs.pyquaticus import Team
from pyquaticus.envs.coach_env import CoachEnv
import time
import os

def env_creator_coach(env_config):
    """
    Creates and returns a new CoachEnv. The underlying PyQuaticusEnv is created with render_mode set to "human"
    so that you can see the simulation while training.
    """
    from pyquaticus import pyquaticus_v0
    # Copy the standard config and update it.
    config_dict = config_dict_std.copy()
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True
    # For a 3v3 game, team_size=3 creates 6 agents.
    underlying_env = pyquaticus_v0.PyQuaticusEnv(
        config_dict=config_dict,
        render_mode="human",  # Change here so the environment renders.
        reward_config=None,
        team_size=3
    )
    # Build the agent_team_map.
    agent_team_map = {agent_id: player.team for agent_id, player in underlying_env.players.items()}
    # Create and return the CoachEnv.
    coach_env = CoachEnv(
        team=env_config["team"],
        env_creator=lambda: underlying_env,  # In production you might want a fresh env each call.
        agent_team_map=agent_team_map
    )
    return coach_env

# Register the environment with RLlib.
register_env("coach_env", env_creator_coach)

def train_coach():
    # Set the configuration for the coach environment.
    env_config = {"team": Team.RED_TEAM}
    config = PPOConfig().environment(env="coach_env", env_config=env_config).framework("torch")
    algo = config.build_algo()

    # Create directory for saving checkpoints.
    checkpoint_dir = os.path.abspath("./ray_test")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop.
    for i in range(10):
        print(f"Training Iteration: {i}")
        start_time = time.time()
        results = algo.train()
        end_time = time.time()
        print(f"Iteration {i} complete. Time Taken: {end_time - start_time:.2f} seconds")

        # Optionally, if your underlying env supports rendering, call its render method.
        # (This may require that your environment's render window is not closed automatically.)
        # underlying_env = algo.workers.local_worker().env
        # underlying_env.render()

        # Save a checkpoint every 5 iterations.
        if i % 5 == 0:
            cp_path = os.path.join(checkpoint_dir, f"iter_{i}")
            os.makedirs(cp_path, exist_ok=True)
            saved_checkpoint = algo.save(cp_path)
            print(f"Checkpoint saved at: {saved_checkpoint}")

if __name__ == '__main__':
    train_coach()
