# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:12:25 2020

@author: Kevin
"""

import numpy as np
import random


class TigerTransition():
    def __init__(self):
        self.transitionMatrix = {
            (2, 0): 0,
            (2, 1): 1,
            (0, 0): 0 if random.random() > 0.5 else 1,

            (0, 1): 0 if random.random() > 0.5 else 1,

            (1, 0): 0 if random.random() > 0.5 else 1,

            (1, 1): 0 if random.random() > 0.5 else 1

        }

    def __call__(self, state, action):
        nextStateProb = self.transitionMatrix[(action, state)]
        return nextStateProb


class TigerReward():
    def __init__(self, rewardParam):
        self.rewardMatrix = {
            (2, 0): rewardParam['listen_cost'],
            (2, 1): rewardParam['listen_cost'],

            (0, 0): rewardParam['open_incorrect_cost'],
            (0, 1): rewardParam['open_correct_reward'],

            (1, 0): rewardParam['open_correct_reward'],
            (1, 1): rewardParam['open_incorrect_cost']
        }

    def __call__(self, state, action, sPrime):
        rewardFixed = self.rewardMatrix.get((action, state), 0.0)
        return rewardFixed


class TigerObservation():
    def __init__(self, observationParam):
        self.observationMatrix = {

            (2, 0): 0 if random.random() < observationParam['obs_correct_prob'] else 1,
            (2, 1): 1 if random.random() < observationParam['obs_correct_prob'] else 0,

            (0, 0): 2,
            (0, 1): 2,
            (1, 0): 2,
            (1, 1): 2
        }

    def __call__(self, state, action):
        observation = self.observationMatrix.get((action, state), 0)
        return observation


class TigerEnv(object):

    def __init__(self):

        rewardParam = {'listen_cost': -1, 'open_incorrect_cost': -100, 'open_correct_reward': 10}
        self.rewardFunction = TigerReward(rewardParam)

        observationParam = {'obs_correct_prob': 0.85, 'obs_incorrect_prob': 0.15}
        self.observationFunction = TigerObservation(observationParam)

        self.transitionFunction = TigerTransition()

        stateSpace = ['tiger-left', 'tiger-right']
        observationSpace = ['tiger-left', 'tiger-right', 'Nothing']
        actionSpace = ['open-left', 'open-right', 'listen']

        self.terminal = False

    def getInitialState(self):
        return 0 if random.random() < 0.5 else 1

    def transition(self, state, action):
        return self.transitionFunction(state, action)

    def reward(self, state, action, sprime):
        reward = self.rewardFunction(state, action, sprime)
        if reward == -100 or reward == 10:
            self.terminal = True
        else:
            self.terminal = False
        return reward

    def observation(self, state, action):
        return self.observationFunction(state, action)

    def isterminal(self, state):
        return self.terminal
