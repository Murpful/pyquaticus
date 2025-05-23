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
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 2v2 PyQuaticus environment')
    parser.add_argument('policy_one', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-0-policy')
    parser.add_argument('policy_two', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-1-policy') 
    parser.add_argument('policy_three', help='Please enter the path to the model you would like to load in Ex. ./ray_test/checkpoint_00001/policies/agent-2-policy')
    reward_config = {}
    args = parser.parse_args()
    config_dict = config_dict_std
    config_dict['sim_speedup_factor'] = 20
    config_dict['max_score'] = 3
    config_dict['max_time']=500
    config_dict['tagging_cooldown'] = 60
    config_dict['tag_on_oob']=True


    #Create Environment
    env = pyquaticus_v0.PyQuaticusEnv(config_dict=config_dict,render_mode='human',reward_config=reward_config, team_size=3)
    
    obs,info = env.reset(return_info=True)
    Attack0 = BaseAttacker('agent_3', Team.RED_TEAM, env, mode='easy')
    Deffend0 = BaseDefender('agent_3', Team.RED_TEAM, env, mode='easy')
    Attack1 = BaseAttacker('agent_4', Team.RED_TEAM, env, mode='easy')
    Deffend1 = BaseDefender('agent_4', Team.RED_TEAM, env, mode='easy')
    Attack2 = BaseAttacker('agent_5', Team.RED_TEAM, env, mode='easy')
    Deffend2 = BaseDefender('agent_5', Team.RED_TEAM, env, mode='easy')
    #Ex. Load in Heurisitc
    #H_one = BaseDefender('agent_0', Team.RED_TEAM, mode='easy')
    #H_two = BaseAttacker('agent_1', Team.RED_TEAM, mode='easy')
    #Load in learned policies
    policy_one_dict = Policy.from_checkpoint(os.path.abspath(args.policy_one))
    policy_two_dict = Policy.from_checkpoint(os.path.abspath(args.policy_two))
    policy_three_dict = Policy.from_checkpoint(os.path.abspath(args.policy_three))
    policy_one = policy_one_dict['agent-0-policy']
    policy_two = policy_two_dict['agent-1-policy']
    policy_three = policy_three_dict['agent-2-policy']
    step = 0
    max_step = 2500

    while True:
        new_obs = {}
        #Get Unnormalized Observation for heuristic agents (H_one, and H_two)
        for k in obs:
            new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])

        #Get learning agent action from policy
        zero = policy_one.compute_single_action(obs['agent_0'])[0]
        one = policy_two.compute_single_action(obs['agent_1'])[0]
        two = policy_three.compute_single_action(obs['agent_2'])[0]
        zeroDeffend = Deffend0.compute_action(new_obs, info)
        oneAttack = Attack1.compute_action(new_obs, info)
        twoAttack = Attack2.compute_action(new_obs, info)
        #Ex. Compute Heuristic agent actons
        #two = H_one.compute_action(new_obs)
        #three = H_two.compute_action(new_obs)
        
        #Step the environment
        #Opponents Don't Move:
        obs, reward, term, trunc, info = env.step({'agent_0':zero,'agent_1':one, 'agent_2':two, 'agent_3':zeroDeffend, 'agent_4':oneAttack, 'agent_5':twoAttack})
        k =  list(term.keys())
        if step >= max_step:
            break
        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            obs,_ = env.reset()
    env.close()