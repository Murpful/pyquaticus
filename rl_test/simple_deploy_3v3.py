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

import argparse
import gymnasium as gym
import numpy as np
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPOTF1Policy, PPOTorchPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec
import os
from pyquaticus.base_policies.base_policies import DefendGen, AttackGen
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy
from pyquaticus.config import config_dict_std
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import pyquaticus.utils.rewards as rew

RENDER_MODE = 'human'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 3v3 PyQuaticus environment')
    reward_config = {'agent_0':rew.example_reward, 'agent_1':rew.example_reward, 'agent_2':rew.example_reward}
    args = parser.parse_args()
    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 4
    config_dict['max_score'] = 3
    config_dict['max_time']=240
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob']=True


    #Create Environment
    env = pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict,render_mode='human',reward_config=reward_config, team_size=3)
    
    obs,info = env.reset(return_info=True)
    
    #Ex. Load in Heurisitc
    Attack0 = BaseAttacker('agent_0', Team.RED_TEAM, env, mode='easy')
    Deffend0 = BaseDefender('agent_0', Team.RED_TEAM, env, mode='easy')
    Attack1 = BaseAttacker('agent_1', Team.RED_TEAM, env, mode='easy')
    Deffend1 = BaseDefender('agent_1', Team.RED_TEAM, env, mode='easy')
    Attack2 = BaseAttacker('agent_2', Team.RED_TEAM, env, mode='easy')
    Deffend2 = BaseDefender('agent_2', Team.RED_TEAM, env, mode='easy')
    Attack3 = BaseAttacker('agent_3', Team.BLUE_TEAM, env, mode='easy')
    Deffend3 = BaseDefender('agent_3', Team.BLUE_TEAM, env, mode='easy')
    Attack4 = BaseAttacker('agent_4', Team.BLUE_TEAM, env, mode='easy')
    Deffend4 = BaseDefender('agent_4', Team.BLUE_TEAM, env, mode='easy')
    Attack5 = BaseAttacker('agent_5', Team.BLUE_TEAM, env, mode='easy')
    Deffend5 = BaseDefender('agent_5', Team.BLUE_TEAM, env, mode='easy')

    step = 0
    max_step = 2500

    while True:
        new_obs = {}
        #Get Unnormalized Observation for heuristic agents (H_one, and H_two)
        for k in obs:
            new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])


        zeroDeffend = Deffend0.compute_action(new_obs, info)
        oneDeffend = Deffend1.compute_action(new_obs, info)
        twoDeffend = Deffend2.compute_action(new_obs, info)
        threeDeffend = Deffend3.compute_action(new_obs, info)
        fourDeffend = Deffend4.compute_action(new_obs, info)
        fiveDeffend = Deffend5.compute_action(new_obs, info)
        zeroAttack = Attack0.compute_action(new_obs, info)
        oneAttack = Attack1.compute_action(new_obs, info)
        twoAttack = Attack2.compute_action(new_obs, info)
        threeAttack = Attack3.compute_action(new_obs, info)
        fourAttack = Attack4.compute_action(new_obs, info)
        fiveAttack = Attack5.compute_action(new_obs, info)
        print("Step: ")
        print(step)
        #Step the environment
        if step >= max_step/2:
            obs, reward, term, trunc, info = env.step({'agent_0':zeroDeffend,'agent_1':oneDeffend, 'agent_2':twoDeffend, 'agent_3':threeDeffend, 'agent_4':fourDeffend, 'agent_5':fiveDeffend})
        else:
            obs, reward, term, trunc, info = env.step({'agent_0':zeroAttack,'agent_1':oneAttack, 'agent_2':twoAttack, 'agent_3':threeAttack, 'agent_4':fourAttack, 'agent_5':fiveAttack})
        k =  list(term.keys())
        
        if step >= max_step:
            break
        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            obs,info = env.reset()
    env.close()


