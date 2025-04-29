# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

"""
#Configureable Rewards
    # -- NOTE --
    #   All headings are in nautical format
    #                 0
    #                 |
    #          270 -- . -- 90
    #                 |
    #                180
    #
    # This can be converted the standard heading format that is counterclockwise
    # by using the heading_angle_conversion(deg) function found in utils.py
    #
    #
    ## Each custom reward function should have the following arguments ##
    Args:
        agent_id (int): ID of the agent we are computing the reward for
        team (Team): team of the agent we are computing the reward for
        agents (list): list of agent ID's (this is used to map agent_id's to agent indices and viceversa)
        agent_inds_of_team (dict): mapping from team to agent indices of that team
        state (dict):
            'agent_position' (array): list of agent positions (indexed in the order of agents list)

                        Ex. Usage: Get agent's current position
                        agent_id = 'agent_1'
                        position = state['agent_position'][agents.index(agent_id)]

            'prev_agent_position' (array): list of agent positions (indexed in the order of agents list) at the previous timestep

                        Ex. Usage: Get agent's previous position
                        agent_id = 'agent_1'
                        prev_position = state['prev_agent_position'][agents.index(agent_id)]

            'agent_speed' (array): list of agent speeds (indexed in the order of agents list)

                        Ex. Usage: Get agent's speed
                        agent_id = 'agent_1'
                        speed = state

            'agent_heading' (array): list of agent headings (indexed in the order of agents list)

                        Ex. Usage: Get agent's heading
                        agent_id = 'agent_1'
                        heading = state['agent_heading'][agents.index(agent_id)]

            'agent_on_sides' (array): list of booleans (indexed in the order of agents list) where True means the
                                      agent is on its own side, and False means the agent is not on its own side

                        Ex. Usage: Check if agent is on its own side
                        agent_id = 'agent_1'
                        on_own_side = state['agent_on_sides'][agents.index(agent_id)]

            'agent_oob' (array): list of booleans (indexed in the order of agents list) where True means the
                                 agent is out-of-bounds (OOB), and False means the agent is not out-of-bounds
                        
                        Ex. Usage: Check if agent is out-of-bounds
                        agent_id = 'agent_1'
                        num_oob = state['agent_oob'][agents.index(agent_id)]
            
            'agent_has_flag' (array): list of booleans (indexed in the order of agents list) where True means the
                                     agent has a flag, and False means the agent does not have a flag

                        Ex. Usage: Check if agent has a flag
                        agent_id = 'agent_1'
                        has_flag = state['agent_has_flag'][agents.index(agent_id)]

            'agent_is_tagged' (array): list of booleans (indexed in the order of agents list) where True means
                                       the agent is tagged, and False means the agent is not tagged

                        Ex. Usage: Check if agent is tagged
                        agent_id = 'agent_1'
                        is_tagged = state['agent_is_tagged'][agents.index(agent_id)]

            'agent_made_tag' (array): list (indexed in the order of agents list) where the value at an entry is the index of a different
                                     agent which the agent at the given index has tagged at the current timestep, otherwise None

                        Ex. Usage: Check if agent has tagged an agent
                        agent_id = 'agent_1'
                        tagged_opponent_idx = state['agent_made_tag'][agents.index(agent_id)]

            'agent_tagging_cooldown' (array): current agent tagging cooldowns (indexed in the order of agents list)
                        Note: agent is able to tag when this value is equal to tagging_cooldown
    
                        Ex. Usage: Get agent's current tagging cooldown
                        agent_id = 'agent_1'
                        cooldown = self.state['agent_tagging_cooldown'][agents.index(agent_id)]

            'dist_bearing_to_obstacles' (dict): For each agent in game list out distances and bearings
                                                to all obstacles in game in order of obstacles list

            'flag_home' (array): list of flag homes (indexed by team number)

            'flag_position' (array): list of flag homes (indexed by team number)

            'flag_taken' (array): list of booleans (indexed by team number) where True means the team's flag
                                  is taken (picked up by an opponent), and False means the flag is not taken 

            'team_has_flag' (array): list of booleans (indexed by team number) where True means an agent of the
                                     team has a flag, and False means that no agents are in possesion of a flag

            'captures' (array): list of total captures made by each team (indexed by team number)

            'tags' (array): list of total tags made by each team (indexed by team number)

            'grabs' (array): list of total flag grabs made by each team (indexed by team number)

            'agent_collisions' (array): list of total agent collisions  for each agent (indexed in the order of agents list)

            'agent_dynamics' (array): list of dictionaries containing agent-specific dynamics information (state attribute of a dynamics class - see dynamics.py)

            ######################################################################################
            ##### The following keys will exist in the state dictionary if lidar_obs is True #####
                'lidar_labels' (dict):

                'lidar_labels' (dict):

                'lidar_labels' (dict):
            ######################################################################################
            
            'obs_hist_buffer' (dict): Observation history buffer where the keys are agent_id's and values are the agents' observations

            'global_state_hist_buffer' (array): Global state history buffer

        prev_state (dict): Contains the state information from the previous step

        env_size (array): field dimensions [horizontal, vertical]

        agent_radii (array): list of agent radii (indexed in the order of agents list)

        catch_radius (float): tag and flag grab radius

        scrimmage_coords (array): endpoints [x,y] of the scrimmage line

        max_speeds (list): list of agent max speeds (indexed in the order of agents list)

        tagging_cooldown (float): tagging cooldown time
"""

import math
import numpy

from pyquaticus.structs import Team
from pyquaticus.utils.utils import *

### Example Reward Funtion ###
def example_reward(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    return 0.0
### Example Reward Funtion ###
def coach(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    return 0.0

def caps_and_grabs(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    reward = 0.0
    prev_num_oob = state['agent_oob'][agents.index(agent_id)]
    num_oob = state['agent_oob'][agents.index(agent_id)]
    if num_oob > prev_num_oob:
        reward += -1.0
    for t in state['grabs']:
        prev_num_grabs = state['grabs'][t]
        num_grabs = state['grabs'][t]
        if num_grabs > prev_num_grabs:
            reward += 0.25 if t == team else -0.25

        prev_num_caps = state['captures'][t]
        num_caps = state['captures'][t]
        if num_caps > prev_num_caps:
            reward += 1.0 if t == team else -1.0

    return reward

### Add Custom Reward Functions Here ###
def control(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    idx = agents.index(agent_id)
    reward = 0.0

    # 1) Penalty for leaving bounds
    if not prev_state['agent_oob'][idx] and state['agent_oob'][idx]:
        reward -= 1.0

    # 2) Incentivize tagging an opponent
    if state['agent_made_tag'][idx] is not None:
        reward += 0.5

    # 3) Punish being tagged
    if not prev_state['agent_is_tagged'][idx] and state['agent_is_tagged'][idx]:
        reward -= 0.5

    # 4) Incentivize grabbing the flag
    if not prev_state['agent_has_flag'][idx] and state['agent_has_flag'][idx]:
        reward += 0.25

    # 5) Incentivize *this agent’s* capture
    team_idx     = team.value
    prev_caps    = prev_state['captures'][team_idx]
    caps_now     = state    ['captures'][team_idx]
    had_flag_before = prev_state['agent_has_flag'][idx]
    has_flag_now    = state    ['agent_has_flag'][idx]

    if had_flag_before and not has_flag_now and caps_now > prev_caps:
        reward += 1.0

    return reward

def balance(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    """
    - Individual penalties:
        * –1.0 for going out of bounds
        * –0.5 for being tagged
    - Team successes (tag, grab, capture) are pooled and split equally.
    """
    idx       = agents.index(agent_id)
    reward    = 0.0
    team_idx  = team.value
    teammates = agent_inds_of_team[team]
    n_team    = len(teammates)

    # 1) Individual penalty: left bounds this step?
    if not prev_state['agent_oob'][idx] and state['agent_oob'][idx]:
        reward -= 1.0

    # 2) Individual penalty: got tagged this step?
    if not prev_state['agent_is_tagged'][idx] and state['agent_is_tagged'][idx]:
        reward -= 0.5

    # 3) Team‐wide tagging reward
    prev_team_tags = prev_state['tags'][team_idx]
    curr_team_tags = state   ['tags'][team_idx]
    delta_tags     = curr_team_tags - prev_team_tags
    if delta_tags > 0:
        reward += (0.5 * delta_tags) / n_team

    # 4) Team‐wide flag‐grab reward
    prev_team_grabs = prev_state['grabs'][team_idx]
    curr_team_grabs = state   ['grabs'][team_idx]
    delta_grabs     = curr_team_grabs - prev_team_grabs
    if delta_grabs > 0:
        reward += (0.25 * delta_grabs) / n_team

    # 5) Team‐wide capture reward
    prev_team_caps = prev_state['captures'][team_idx]
    curr_team_caps = state   ['captures'][team_idx]
    delta_caps     = curr_team_caps - prev_team_caps
    if delta_caps > 0:
        reward += (1.0 * delta_caps) / n_team

    return reward

def team(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    """
    - Individual penalties (same as before):
        * –1.0 for leaving bounds
        * –0.5 for being tagged
    - Personal successes get a modest bonus; teammate successes get a larger bonus.
    """
    idx       = agents.index(agent_id)
    reward    = 0.0
    t_idx     = team.value

    # --- 1) Individual penalties ---
    if not prev_state['agent_oob'][idx] and state['agent_oob'][idx]:
        reward -= 1.0
    if not prev_state['agent_is_tagged'][idx] and state['agent_is_tagged'][idx]:
        reward -= 0.5

    # --- weight settings (tune these as you like) ---
    SELF_TAG_W    = 0.5
    TEAM_TAG_W    = 1.0

    SELF_GRAB_W   = 0.25
    TEAM_GRAB_W   = 0.5

    SELF_CAP_W    = 1.0
    TEAM_CAP_W    = 2.0

    # --- 2) Tagging rewards ---
    # total team tags this step
    delta_tags = state['tags'][t_idx] - prev_state['tags'][t_idx]
    # did I tag someone?
    did_personal_tag = (state['agent_made_tag'][idx] is not None)
    if did_personal_tag:
        reward += SELF_TAG_W
    # teammates’ tags = total minus mine
    teammate_tags = max(0, delta_tags - (1 if did_personal_tag else 0))
    if teammate_tags:
        reward += TEAM_TAG_W * teammate_tags

    # --- 3) Grab rewards ---
    had_flag_before = prev_state['agent_has_flag'][idx]
    has_flag_now    = state    ['agent_has_flag'][idx]
    delta_grabs     = state['grabs'][t_idx] - prev_state['grabs'][t_idx]
    did_personal_grab = (not had_flag_before and has_flag_now)
    if did_personal_grab:
        reward += SELF_GRAB_W
    teammate_grabs = max(0, delta_grabs - (1 if did_personal_grab else 0))
    if teammate_grabs:
        reward += TEAM_GRAB_W * teammate_grabs

    # --- 4) Capture rewards ---
    prev_caps = prev_state['captures'][t_idx]
    curr_caps = state    ['captures'][t_idx]
    delta_caps = curr_caps - prev_caps

    # detect personal capture (agent had flag, then dropped it home)
    did_personal_cap = (had_flag_before and not has_flag_now and delta_caps > 0)
    if did_personal_cap:
        reward += SELF_CAP_W
    teammate_caps = max(0, delta_caps - (1 if did_personal_cap else 0))
    if teammate_caps:
        reward += TEAM_CAP_W * teammate_caps

    return reward
def controla(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    idx = agents.index(agent_id)
    reward = 0.0

    # --- Scaled constants ---
    PENALTY_OOB = -1.0
    PENALTY_TAGGED = -0.5

    REWARD_TAG = 0.5
    REWARD_GRAB = 0.3
    REWARD_CAP = 1.0

    MOVE_REWARD_NO_FLAG = 0.01
    MOVE_REWARD_WITH_FLAG = 0.02

    TIME_STEP_PENALTY = -0.01

    # --- 1) Penalty for leaving bounds ---
    if not prev_state['agent_oob'][idx] and state['agent_oob'][idx]:
        reward += PENALTY_OOB

    # --- 2) Reward for tagging an opponent ---
    if state['agent_made_tag'][idx] is not None:
        reward += REWARD_TAG

    # --- 3) Penalty for being tagged ---
    if not prev_state['agent_is_tagged'][idx] and state['agent_is_tagged'][idx]:
        reward += PENALTY_TAGGED

    # --- 4) Reward for grabbing the flag ---
    if not prev_state['agent_has_flag'][idx] and state['agent_has_flag'][idx]:
        reward += REWARD_GRAB

    # --- 5) Reward for personal capture (if carried flag and now captured) ---
    team_idx = team.value
    prev_caps = prev_state['captures'][team_idx]
    curr_caps = state['captures'][team_idx]
    had_flag_before = prev_state['agent_has_flag'][idx]
    has_flag_now = state['agent_has_flag'][idx]

    if had_flag_before and not has_flag_now and curr_caps > prev_caps:
        reward += REWARD_CAP

    # --- 6) Movement incentives ---
    curr_pos = np.array(state['agent_position'][idx])
    prev_pos = np.array(state['prev_agent_position'][idx])

    opponent_flag_pos = np.array(state['flag_home'][1 - team_idx])
    home_pos = np.array(state['flag_home'][team_idx])

    if not state['agent_has_flag'][idx]:
        # Move toward enemy flag
        d_prev = np.linalg.norm(opponent_flag_pos - prev_pos)
        d_curr = np.linalg.norm(opponent_flag_pos - curr_pos)
        reward += MOVE_REWARD_NO_FLAG * (d_prev - d_curr)
    else:
        # Move toward home
        d_prev = np.linalg.norm(home_pos - prev_pos)
        d_curr = np.linalg.norm(home_pos - curr_pos)
        reward += MOVE_REWARD_WITH_FLAG * (d_prev - d_curr)

    # --- 7) Small penalty for every step (optional) ---
    reward += TIME_STEP_PENALTY

    return reward



def balancea(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    idx = agents.index(agent_id)
    reward = 0.0
    team_idx = team.value
    teammates = agent_inds_of_team[team]
    n_team = len(teammates)

    # --- Scaled constants ---
    PENALTY_OOB = -1.0
    PENALTY_TAGGED = -0.5

    SELF_TAG_REWARD = 0.5
    TEAM_TAG_REWARD = 0.25

    SELF_GRAB_REWARD = 0.5
    TEAM_GRAB_REWARD = 0.25

    SELF_CAP_REWARD = 1.0
    TEAM_CAP_REWARD = 0.5

    MOVE_REWARD_NO_FLAG = 0.01
    MOVE_REWARD_WITH_FLAG = 0.02

    TIME_STEP_PENALTY = -0.01

    # --- 1) Individual penalties ---
    if not prev_state['agent_oob'][idx] and state['agent_oob'][idx]:
        reward += PENALTY_OOB

    if not prev_state['agent_is_tagged'][idx] and state['agent_is_tagged'][idx]:
        reward += PENALTY_TAGGED

    # --- 2) Tags ---
    delta_tags = state['tags'][team_idx] - prev_state['tags'][team_idx]
    did_personal_tag = (state['agent_made_tag'][idx] is not None)

    if delta_tags > 0:
        if did_personal_tag:
            reward += SELF_TAG_REWARD
        else:
            reward += TEAM_TAG_REWARD

    # --- 3) Flag grabs ---
    delta_grabs = state['grabs'][team_idx] - prev_state['grabs'][team_idx]
    had_flag_before = prev_state['agent_has_flag'][idx]
    has_flag_now = state['agent_has_flag'][idx]
    did_personal_grab = (not had_flag_before and has_flag_now)

    if delta_grabs > 0:
        if did_personal_grab:
            reward += SELF_GRAB_REWARD
        else:
            reward += TEAM_GRAB_REWARD

    # --- 4) Captures ---
    delta_caps = state['captures'][team_idx] - prev_state['captures'][team_idx]
    did_personal_cap = (had_flag_before and not has_flag_now and delta_caps > 0)

    if delta_caps > 0:
        if did_personal_cap:
            reward += SELF_CAP_REWARD
        else:
            reward += TEAM_CAP_REWARD

    # --- 5) Movement rewards ---
    curr_pos = np.array(state['agent_position'][idx])
    prev_pos = np.array(state['prev_agent_position'][idx])

    opponent_flag_pos = np.array(state['flag_home'][1 - team_idx])
    home_pos = np.array(state['flag_home'][team_idx])

    if not state['agent_has_flag'][idx]:
        d_prev = np.linalg.norm(opponent_flag_pos - prev_pos)
        d_curr = np.linalg.norm(opponent_flag_pos - curr_pos)
        reward += MOVE_REWARD_NO_FLAG * (d_prev - d_curr)
    else:
        d_prev = np.linalg.norm(home_pos - prev_pos)
        d_curr = np.linalg.norm(home_pos - curr_pos)
        reward += MOVE_REWARD_WITH_FLAG * (d_prev - d_curr)

    # --- 6) Time penalty ---
    reward += TIME_STEP_PENALTY

    return reward




def teama(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    idx = agents.index(agent_id)
    reward = 0.0
    t_idx = team.value
    teammates = agent_inds_of_team[team]
    n_team = len(teammates)

    # --- Scaled constants ---
    PENALTY_OOB = -1.0
    PENALTY_TAGGED = -0.5

    TEAM_TAG_REWARD = 0.3
    TEAM_GRAB_REWARD = 0.5
    TEAM_CAP_REWARD = 1.0

    MOVE_REWARD_NO_FLAG = 0.01
    MOVE_REWARD_WITH_FLAG = 0.02

    TIME_STEP_PENALTY = -0.01

    # --- 1) Individual penalties ---
    if not prev_state['agent_oob'][idx] and state['agent_oob'][idx]:
        reward += PENALTY_OOB

    if not prev_state['agent_is_tagged'][idx] and state['agent_is_tagged'][idx]:
        reward += PENALTY_TAGGED

    # --- 2) Tags ---
    delta_tags = state['tags'][t_idx] - prev_state['tags'][t_idx]
    if delta_tags > 0:
        reward += (TEAM_TAG_REWARD * delta_tags) / n_team

    # --- 3) Flag grabs ---
    delta_grabs = state['grabs'][t_idx] - prev_state['grabs'][t_idx]
    if delta_grabs > 0:
        reward += (TEAM_GRAB_REWARD * delta_grabs) / n_team

    # --- 4) Captures ---
    delta_caps = state['captures'][t_idx] - prev_state['captures'][t_idx]
    if delta_caps > 0:
        reward += (TEAM_CAP_REWARD * delta_caps) / n_team

    # --- 5) Movement rewards ---
    curr_pos = np.array(state['agent_position'][idx])
    prev_pos = np.array(state['prev_agent_position'][idx])

    opponent_flag_pos = np.array(state['flag_home'][1 - t_idx])
    home_pos = np.array(state['flag_home'][t_idx])

    if not state['agent_has_flag'][idx]:
        d_prev = np.linalg.norm(opponent_flag_pos - prev_pos)
        d_curr = np.linalg.norm(opponent_flag_pos - curr_pos)
        reward += MOVE_REWARD_NO_FLAG * (d_prev - d_curr)
    else:
        d_prev = np.linalg.norm(home_pos - prev_pos)
        d_curr = np.linalg.norm(home_pos - curr_pos)
        reward += MOVE_REWARD_WITH_FLAG * (d_prev - d_curr)

    # --- 6) Time penalty ---
    reward += TIME_STEP_PENALTY

    return reward

