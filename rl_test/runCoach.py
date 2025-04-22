#!/usr/bin/env python
import numpy as np
from pyquaticus.envs.pyquaticus import Team
from pyquaticus.couchquaticus import CoachAgent
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender

# Helper function to aggregate individual agent observations into a team observation.
def aggregate_team_obs(new_obs, team, agent_team_map):
    team_obs = [new_obs[agent_id] for agent_id in new_obs if agent_team_map[agent_id] == team]
    if len(team_obs) == 0:
        raise ValueError("No observations found for the specified team.")
    aggregated_obs = np.mean(np.array(team_obs), axis=0)
    return aggregated_obs

def main():
    # Import environment and config.
    from pyquaticus import pyquaticus_v0
    from pyquaticus.config import config_dict_std

    # Modify configuration as needed.
    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time'] = 240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob'] = True

    # Create a 3v3 environment (team_size=3 implies 6 agents total).
    env = pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict, render_mode='human',
                                      reward_config=None, team_size=3)
    obs, info = env.reset(return_info=True)

    # Build a mapping from agent id to its team.
    agent_team_map = {}
    for agent_id, player in env.players.items():
        agent_team_map[agent_id] = player.team

    # Create a coach agent for each team.
    red_coach = CoachAgent(Team.RED_TEAM)
    blue_coach = CoachAgent(Team.BLUE_TEAM)

    # Instantiate heuristic agents for each team.
    # Red team agents: 'agent_0', 'agent_1', 'agent_2'
    Attack0 = BaseAttacker('agent_0', Team.RED_TEAM, env, mode='easy')
    Defend0 = BaseDefender('agent_0', Team.RED_TEAM, env, mode='easy')
    Attack1 = BaseAttacker('agent_1', Team.RED_TEAM, env, mode='easy')
    Defend1 = BaseDefender('agent_1', Team.RED_TEAM, env, mode='easy')
    Attack2 = BaseAttacker('agent_2', Team.RED_TEAM, env, mode='easy')
    Defend2 = BaseDefender('agent_2', Team.RED_TEAM, env, mode='easy')

    # Blue team agents: 'agent_3', 'agent_4', 'agent_5'
    Attack3 = BaseAttacker('agent_3', Team.BLUE_TEAM, env, mode='easy')
    Defend3 = BaseDefender('agent_3', Team.BLUE_TEAM, env, mode='easy')
    Attack4 = BaseAttacker('agent_4', Team.BLUE_TEAM, env, mode='easy')
    Defend4 = BaseDefender('agent_4', Team.BLUE_TEAM, env, mode='easy')
    Attack5 = BaseAttacker('agent_5', Team.BLUE_TEAM, env, mode='easy')
    Defend5 = BaseDefender('agent_5', Team.BLUE_TEAM, env, mode='easy')

    step = 0
    max_steps = 2500
    while step < max_steps:
        # Unnormalize each agent's observation.
        new_obs = {}
        for agent_id in obs:
            new_obs[agent_id] = env.agent_obs_normalizer.unnormalized(obs[agent_id])
        
        # Instead of aggregating all observations, simply pick one agent per team.
        red_team_obs = new_obs['agent_0']   # For Team RED
        blue_team_obs = new_obs['agent_3']  # For Team BLUE

        
        # Get high-level decisions from the coach agents.
        red_decision = red_coach.compute_action(red_team_obs)
        blue_decision = blue_coach.compute_action(blue_team_obs)
        
        # Prepare actions based on the coach decisions.
        actions = {}
        if red_decision == 0:
            # Red team attack
            actions['agent_0'] = Attack0.compute_action(new_obs, info)
            actions['agent_1'] = Attack1.compute_action(new_obs, info)
            actions['agent_2'] = Attack2.compute_action(new_obs, info)
        else:
            # Red team defend
            actions['agent_0'] = Defend0.compute_action(new_obs, info)
            actions['agent_1'] = Defend1.compute_action(new_obs, info)
            actions['agent_2'] = Defend2.compute_action(new_obs, info)
        
        if blue_decision == 0:
            # Blue team attack
            actions['agent_3'] = Attack3.compute_action(new_obs, info)
            actions['agent_4'] = Attack4.compute_action(new_obs, info)
            actions['agent_5'] = Attack5.compute_action(new_obs, info)
        else:
            # Blue team defend
            actions['agent_3'] = Defend3.compute_action(new_obs, info)
            actions['agent_4'] = Defend4.compute_action(new_obs, info)
            actions['agent_5'] = Defend5.compute_action(new_obs, info)
        
        # Step the environment.
        obs, reward, term, trunc, info = env.step(actions)
        step += 1
        
        # Reset environment if terminal or truncated.
        if term.get('__all__', False) or trunc.get('__all__', False):
            obs, info = env.reset(return_info=True)

if __name__ == '__main__':
    main()
