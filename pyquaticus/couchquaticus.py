import numpy as np
import gymnasium as gym
from pyquaticus.envs.pyquaticus import Team  # adjust the import as needed

def flatten_obs(obs):
    """
    Recursively flatten an observation (dict, list, tuple, or np.ndarray) into a flat list of floats.
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
        except Exception as e:
            print("Warning: could not convert", obs, "to float:", e)
            return [0.0]

def aggregate_team_obs(new_obs, team, agent_team_map):
    team_obs_list = []
    for agent_id, obs_val in new_obs.items():
        if agent_team_map[agent_id] == team:
            if isinstance(obs_val, dict) or isinstance(obs_val, (list, tuple)):
                flat_list = flatten_obs(obs_val)
                flat_obs = np.array(flat_list, dtype=float)
            else:
                flat_obs = np.array(obs_val, dtype=float)
            team_obs_list.append(flat_obs)
    if len(team_obs_list) == 0:
        raise ValueError("No observations found for the specified team.")
    
    # Debug: Print the shape and data for each flattened observation.
    for i, arr in enumerate(team_obs_list):
        print(f"Agent index {i} flattened shape: {arr.shape}, data: {arr}")
    
    # Check that all flattened arrays have the same shape.
    shapes = [arr.shape for arr in team_obs_list]
    if len(set(shapes)) != 1:
        raise ValueError(f"Inconsistent flattened observation shapes: {shapes}")
    
    aggregated_obs = np.mean(np.array(team_obs_list, dtype=float), axis=0)
    return aggregated_obs










class CoachAgent:
    def __init__(self, team, coach_obs_space=None, coach_action_space=None):
        self.team = team
        # For now, we use a simple discrete action space:
        # 0 for "attack" and 1 for "defend".
        if coach_action_space is None:
            self.coach_action_space = gym.spaces.Discrete(2)
        else:
            self.coach_action_space = coach_action_space
        
        # Optionally store the observation space for later use
        self.coach_obs_space = coach_obs_space

    def compute_action(self, aggregated_obs):
        """
        Given an aggregated observation, compute a high-level decision.
        This dummy implementation simply returns a random decision.
        
        Args:
            aggregated_obs (np.ndarray): Aggregated observation for the team.
            
        Returns:
            int: 0 (attack) or 1 (defend).
        """
        return self.coach_action_space.sample()
if __name__ == '__main__':
    # Create a dummy aggregated observation. Adjust dimensions as needed.
    dummy_aggregated_obs = np.array([5, 5, 5, 5, 5])
    
    # Create coach instances for both teams.
    red_coach = CoachAgent(Team.RED_TEAM)
    blue_coach = CoachAgent(Team.BLUE_TEAM)
    
    # Compute decisions for both teams.
    red_decision = red_coach.compute_action(dummy_aggregated_obs)
    blue_decision = blue_coach.compute_action(dummy_aggregated_obs)
    
    print("Red Coach Decision (0=attack, 1=defend):", red_decision)
    print("Blue Coach Decision (0=attack, 1=defend):", blue_decision)