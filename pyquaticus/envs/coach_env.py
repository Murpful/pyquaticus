import gymnasium as gym
import numpy as np
from pyquaticus.envs.pyquaticus import Team

def get_team_obs_first(new_obs, team, agent_team_map):
    for agent_id, obs in new_obs.items():
        if agent_team_map[agent_id] == team:
            return obs
    raise ValueError("No observation found for the specified team.")

def flatten_obs(obs):
    """
    Recursively flatten an observation (dict, list, tuple, or np.ndarray)
    into a flat list of floats.
    """
    flat_list = []
    if isinstance(obs, dict):
        for key in sorted(obs.keys(), key=lambda x: str(x)):
            flat_list.extend(flatten_obs(obs[key]))
        return flat_list
    elif isinstance(obs, (list, tuple, np.ndarray)):
        for item in obs:
            flat_list.extend(flatten_obs(item))
        return flat_list
    else:
        try:
            return [float(obs)]
        except Exception:
            return [0.0]

class CoachEnv(gym.Env):
    """
    A custom Gym environment for training a coach agent.
    Instead of storing an already-created underlying environment,
    this environment stores an env_creator callable that creates a fresh underlying environment.
    The coach observes the unnormalized observation from the first agent on its team,
    and outputs a discrete action: 0 (attack) or 1 (defend).
    The reward is defined as the sum of rewards obtained by all agents on that team.
    """
    def __init__(self, team, env_creator, agent_team_map):
        super().__init__()
        self.team = team
        self.env_creator = env_creator  # a callable that creates the underlying environment
        self.agent_team_map = agent_team_map
        self.underlying_env = self.env_creator()

        # Use the observation from the first agent as a sample.
        obs, _ = self.underlying_env.reset(return_info=True)
        sample_obs = get_team_obs_first(obs, self.team, self.agent_team_map)
        flat_sample = np.array(flatten_obs(sample_obs), dtype=float)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=flat_sample.shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, **kwargs):
        obs, info = self.underlying_env.reset(return_info=True)
        team_obs = get_team_obs_first(obs, self.team, self.agent_team_map)
        flat_obs = np.array(flatten_obs(team_obs), dtype=float)
        return flat_obs, info

    def step(self, action):
        # For demonstration, we apply the coach action to all team agents.
        chosen_actions = {}
        for agent_id in self.agent_team_map:
            if self.agent_team_map[agent_id] == self.team:
                chosen_actions[agent_id] = action  # dummy mapping
        # Step the underlying environment.
        obs, reward, term, trunc, info = self.underlying_env.step(chosen_actions)
        team_obs = get_team_obs_first(obs, self.team, self.agent_team_map)
        flat_obs = np.array(flatten_obs(team_obs), dtype=float)
        # Compute team reward as the sum of rewards for agents on this team.
        # Example: reward might be sum of agent rewards plus a bonus if team score improved.
        team_agent_rewards = [reward[agent_id] for agent_id in reward if self.agent_team_map[agent_id] == self.team]
        team_reward = np.sum(team_agent_rewards)  
        # Optionally, add extra bonus/penalty here if you have team-level metrics (e.g., changes in flag captures).

        # Use separate flags for terminated and truncated.
        terminated = term.get('__all__', False)
        truncated = trunc.get('__all__', False)
        return flat_obs, team_reward, terminated, truncated, info


    def __getstate__(self):
        """
        Remove the underlying_env from state so that non-pickleable objects are not deep copied.
        """
        state = self.__dict__.copy()
        if 'underlying_env' in state:
            del state['underlying_env']
        return state

    def __setstate__(self, state):
        """
        Recreate the underlying environment from the env_creator after unpickling.
        """
        self.__dict__.update(state)
        self.underlying_env = self.env_creator()
