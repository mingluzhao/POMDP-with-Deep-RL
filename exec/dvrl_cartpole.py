import numpy as np
import torch
import torch.nn as nn
import collections
import gym

from src.dvrl.storage import RolloutStorage
from src.dvrl.utils import toOneHot
from src.dvrl.dvrl import DVRL
from src.dvrl.networks import VRNN_encoding, VRNN_proposal, VRNN_deterministic_transition, VRNN_transition, \
    VRNN_emission
from src.dvrl.policy import Categorical


def setupExperiment(config):
    env = gym.make(config["envName"])
    actionDim = env.action_space.n
    observationDim = 2  # env.observation_space.shape[0]

    encodingNetwork = VRNN_encoding(observationDim, config["hiddenDim"],
                                    config["observationEncodeDim"], actionDim, config["actionEncodeDim"])
    proposalNetwork = VRNN_proposal(config["zDim"], config["hDim"], config["observationEncodeDim"],
                                    config["actionEncodeDim"])
    deterministicTransitionNetwork = VRNN_deterministic_transition(config["zDim"], config["observationEncodeDim"],
                                                                   config["hDim"],
                                                                   config["actionEncodeDim"])
    transitionNetwork = VRNN_transition(config["hDim"], config["zDim"], config["actionEncodeDim"])
    emissionNetwork = VRNN_emission(config["hDim"], config["actionEncodeDim"], config["hiddenDim"], observationDim,
                                    config["observationEncodeDim"])

    particleGru = nn.GRU(config["hDim"] * 2 + 1, config["hDim"], batch_first=True)

    criticLinear = nn.Linear(config["hDim"], 1)

    actionDist = Categorical(config["hDim"], actionDim)

    actorCritic = DVRL(actionDim,
                       observationDim,
                       config["actionEncodeDim"],
                       config["observationEncodeDim"],
                       config["hiddenDim"],
                       config["hDim"],
                       config["zDim"],
                       config["numParticles"],
                       encodingNetwork,
                       proposalNetwork,
                       deterministicTransitionNetwork,
                       transitionNetwork,
                       emissionNetwork,
                       particleGru,
                       criticLinear,
                       actionDist)

    # initialize rolloutstorage
    rolloutStorage = RolloutStorage(config["numStepBeforeTrain"])

    # initialize current_memory
    obs = env.reset()
    obs = obs[::2]
    currentObs = torch.from_numpy(obs).float().unsqueeze(0)

    initStates = actorCritic.newLatentState()
    initRewards = torch.zeros(1, 1)
    initActions = torch.zeros(1, 1)
    currentMemory = {
        'currentObs': currentObs,
        'states': initStates,
        'oneHotActions': toOneHot(
            actionDim,
            initActions),
        'rewards': initRewards
    }
    return env, actorCritic, rolloutStorage, currentMemory


def updateMemory(env, actorCritic, currentMemory, policyReturn, obs, reward, done):
    # Make reward into tensor so we can use it as input to model
    reward = torch.from_numpy(np.expand_dims(np.stack(np.array([reward])), 1)).float()
    # If trajectory ended, create mask to clean reset actions and latent states
    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in [done]])

    # Update current_memory
    currentMemory['currentObs'] = torch.from_numpy(np.array([obs])).float()
    # Create new latent states for new episodes
    # only create new latent state if one episode is done
    currentMemory['states'] = policyReturn.latentState if not done else actorCritic.newLatentState()
    # Set first action to 0 for new episodes
    # Also, if action is discrete, convert it to one-hot vector
    currentMemory['oneHotActions'] = toOneHot(
        env.action_space.n,
        policyReturn.action * masks.type(policyReturn.action.type()))
    currentMemory['rewards'][:] = reward

    return currentMemory, masks, reward


def runOneTimeStep(actorCritic, currentMemory, env, episodeStep):
    # use policy to get action and other stuff
    policyReturn = actorCritic(currentMemory)
    actions = policyReturn.action.detach().squeeze(1).numpy()[0]
    obs, reward, done, info = env.step(actions)

    """unique to cartpole"""
    obs = obs[::2]
    episodeStep += 1
    if done:
        obs = env.reset()
        obs = obs[::2]
        print("score is:{}".format(episodeStep))
        episodeStep = 0
    """unique to cartpole"""

    currentMemory, masks, reward = updateMemory(env, actorCritic, currentMemory, policyReturn, obs, reward, done)

    return policyReturn, currentMemory, masks, reward, episodeStep


def trackValues(trackedValues, policyReturn):
    # track value to calculate loss with respect to targets

    trackedValues['values'].append(policyReturn.valueEstimate)
    trackedValues['actionLogProbs'].append(policyReturn.actionLogProbs)
    trackedValues['distEntropy'].append(policyReturn.distEntropy)
    # For loss function
    trackedValues['totalEncodingLoss'].append(policyReturn.totalEncodingLoss)

    return trackedValues


def main():
    config = {"envName": 'CartPole-v1',
              "actionEncodeDim": 64,
              "observationEncodeDim": 128,
              "hiddenDim": 64,
              "hDim": 256,
              "zDim": 256,
              "numParticles": 15,
              "numStepBeforeTrain": 4,
              "totalTrainStep": 100000
              }

    env, actorCritic, rollouts, currentMemory = setupExperiment(config)
    episodeStep = 0
    # start running
    for j in range(config["totalTrainStep"]):

        trackedValues = collections.defaultdict(lambda: [])

        for step in range(config["numStepBeforeTrain"]):
            policyReturn, currentMemory, masks, reward, episodeStep = runOneTimeStep(actorCritic, currentMemory, env,
                                                                                     episodeStep)
            # Save in rollouts (for loss computation)
            rollouts.insert(step, reward, masks)

            # Track all bunch of stuff and also save intermediate images and stuff
            trackedValues = trackValues(trackedValues, policyReturn)

        actorCritic.learn(rollouts, trackedValues, currentMemory)


if __name__ == '__main__':
    main()

"""
./code/main.py -p with environment.config_file=openaiEnv.yaml environment.name=PongNoFrameskip-v0 algorithm.use_particle_filter=True algorithm.model.h_dim=256 algorithm.multiplier_backprop_length=10 algorithm.particle_filter.num_particles=15 opt.lr=2.0e-04 loss_function.encoding_loss_coef=0.1 rl_setting.num_processes=1 algorithm.model.policy_batch_norm=False
"""
