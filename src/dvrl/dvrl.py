import torch
import torch.nn as nn
import numpy as np
import src.dvrl.aesmc.math as math
from src.dvrl.aesmc.inference import sample_ancestral_index
import src.dvrl.aesmc.state as st
import namedlist
from torch.autograd import Variable
import torch.optim as optim

# Container to return all required values from model
PolicyReturn = namedlist.namedlist('PolicyReturn', [
    ('latentState', None),
    ('valueEstimate', None),
    ('action', None),
    ('actionLogProbs', None),
    ('distEntropy', None),
    ('totalEncodingLoss', None)
])


def sampleUniform(batch_size):
    uniforms = np.random.uniform(size=[batch_size, 1])
    return uniforms


class DVRL(nn.Module):
    def __init__(self,
                 actionDim,
                 observationDim,
                 actionEncodeDim,
                 observationEncodeDim,
                 hiddenDim,
                 hDim,
                 zDim,
                 numParticles,
                 encodingNetwork,
                 proposalNetwork,
                 deterministicTransitionNetwork,
                 transitionNetwork,
                 emissionNetwork,
                 particleGru,
                 criticLinear,
                 actionDist,
                 batchSize=1):
        super().__init__()

        self.actionDim = actionDim
        self.observationDim = observationDim
        self.actionEncodeDim = actionEncodeDim
        self.observationEncodeDim = observationEncodeDim
        self.hDim = hDim
        self.zDim = zDim
        self.hiddenDim = hiddenDim
        self.numParticles = numParticles
        self.batchSize = batchSize
        # Input dimension is hDim + hDim + 1, ok to set output dim = hDim
        self.particleGru = particleGru

        self.encodingNetwork = encodingNetwork
        self.proposalNetwork = proposalNetwork
        self.deterministicTransitionNetwork = deterministicTransitionNetwork
        self.transitionNetwork = transitionNetwork
        self.emissionNetwork = emissionNetwork

        # these two are the final output
        # with the same input, one outputs value the other outputs action
        self.criticLinear = criticLinear
        self.dist = actionDist

        self.optimizer = optim.RMSprop(self.parameters(), 0.0002, eps=1e-05, alpha=0.99)

    # resample particle from index
    def sampleFromDistribution(self, distribution):
        return distribution.sample_reparameterized(
            self.batchSize, self.numParticles)

    # initialize new latent state for new episodes
    def newLatentState(self):
        initialState = st.State(
            h=torch.zeros(self.batchSize, self.numParticles, self.hDim),
            logWeight=torch.zeros(self.batchSize, self.numParticles))

        return initialState

    def encodeObservationAction(self, observation, reward, action):
        obsActUnencode = st.State(
            observation=observation.contiguous().unsqueeze(0),
            action=action.contiguous().unsqueeze(0),
            reward=reward.contiguous().unsqueeze(0))
        # encode observation and action
        obsActEncode = self.encodingNetwork(obsActUnencode)

        return obsActEncode

    def resampleParticles(self, previousLatentState):
        previousLogWeight = previousLatentState.logWeight
        # compute u, choose some particles to replace all particles
        ancestralIndices = sample_ancestral_index(previousLogWeight, sampleUniform)
        # resample based on u
        previousLatentState = previousLatentState.resample(ancestralIndices)

        return previousLatentState

    def computeDistribution(self, previousLatentState, obsActEncode):
        # choose the first timestep observation,action,encoded obs, encoded_Action,etc
        # only one timestep but here to stay with original code
        currentObsActEncode = obsActEncode.index_elements(0)

        proposalDistribution = self.proposalNetwork(
            previousLatentState=previousLatentState,
            obsActEncode=currentObsActEncode
        )
        # sample z  [1, 15, 256]
        latentState = self.sampleFromDistribution(proposalDistribution)

        # Compute deterministic state h and add to the latent state
        latentState = self.deterministicTransitionNetwork(
            previousLatentState=previousLatentState,
            latentState=latentState,
            obsActEncode=currentObsActEncode
        )

        # Compute prior probability over z; to calculate w, work with emission and proposal to calculate w together
        priorDistribution = self.transitionNetwork(
            previousLatentState=previousLatentState,
            obsActEncode=currentObsActEncode
        )

        # Compute probability over observation; this is the decoder
        observationDecodeDistribution = self.emissionNetwork(
            previousLatentState=previousLatentState,
            latentState=latentState,
            obsActEncode=currentObsActEncode
        )

        return latentState, proposalDistribution, priorDistribution, observationDecodeDistribution, currentObsActEncode

    def computeWeightLoss(self, latentState, proposalDistribution, priorDistribution, observationDecodeDistribution,
                          currentObsActEncode):
        emissionLogpdf = observationDecodeDistribution.logpdf(
            currentObsActEncode, self.batchSize, self.numParticles)

        proposalLogpdf = proposalDistribution.logpdf(
            latentState, self.batchSize, self.numParticles)

        transitionLogpdf = priorDistribution.logpdf(
            latentState, self.batchSize, self.numParticles)

        # compute the new logWeight and update the particles
        newLogWeight = transitionLogpdf - proposalLogpdf + emissionLogpdf
        latentState.logWeight = newLogWeight

        # Average (in log space) over particles
        # !!! This is ELBO loss for particle filter !!!!!!!!
        encodingLogli = math.logsumexp(
            newLogWeight, dim=1
        ) - np.log(self.numParticles)

        return latentState, -encodingLogli

    def encode(self, observation, reward, action, previousLatentState):
        # Encode
        obsActEncode = self.encodeObservationAction(observation, reward, action)

        # expand ObsActEncode to # of particles
        # later in particles calculation, every particle needs the info -- so need to expand
        obsActEncode.unsqueeze_and_expand_all_(dim=2, size=self.numParticles)

        # resample particles
        previousLatentState = self.resampleParticles(previousLatentState)

        # compute three distributions in formula for particle weights
        latentState, proposalDistribution, \
        priorDistribution, observationDecodeDistribution, currentObsActEncode = self.computeDistribution(
            previousLatentState, obsActEncode)

        # compute new weight and reconstruction loss
        latentStateNewWeight, elboLoss = self.computeWeightLoss(latentState, proposalDistribution, \
                                                                priorDistribution, observationDecodeDistribution,
                                                                currentObsActEncode)

        return latentStateNewWeight, elboLoss

    def catEncodeParticles(self, latentState):
        batchSize, numParticles, _ = latentState.h.size()

        normalizedLogWeights = math.lognormexp(
            latentState.logWeight,
            dim=1
        )
        particleState = torch.cat([latentState.h,
                                   latentState.encoded_Z,
                                   # add a dimension at the last dim for weight
                                   torch.exp(normalizedLogWeights).unsqueeze(-1)],
                                  dim=2)

        _, encodedParticles = self.particleGru(particleState)

        return encodedParticles[0]

    def forward(self, currentMemory, deterministic=False):
        policyReturn = PolicyReturn()

        # currentMemory = current_observation, previous_action, previous_state

        latentState, totalEncodingLoss = self.encode(
            observation=currentMemory['currentObs'],
            reward=currentMemory['rewards'],
            action=currentMemory['oneHotActions'].detach(),
            previousLatentState=currentMemory['states'],
        )

        encodedState = self.catEncodeParticles(latentState)

        # get v
        valueEstimate = self.criticLinear(encodedState)
        # get action
        action = self.dist.sample(encodedState, deterministic=deterministic)
        # get action log probability and action entropy
        actionLogProbs, distEntropy = self.dist.logprobs_and_entropy(encodedState, action.detach())

        policyReturn.latentState = latentState
        policyReturn.totalEncodingLoss = totalEncodingLoss
        policyReturn.valueEstimate = valueEstimate
        policyReturn.action = action
        policyReturn.actionLogProbs = actionLogProbs
        policyReturn.distEntropy = distEntropy

        # Have to return this policy_return because we need to keep track of all these information, not only action
        return policyReturn

    def learn(self, rolloutstorage, trackedValues, currentMemory, gamma=0.99,
              valueLossCoef=0.5, actionLossCoef=1.0, entropyCoef=0.01, encodingLossCoef=0.1, retainGraph=False):
        # we need rolloutstorage to compute target
        # we need trackedValues to compute loss with respect to target
        # we need current memory to bootstrap a nextvalue to propagate to previous target values

        with torch.no_grad():
            policyReturn = self.forward(currentMemory=currentMemory)
            # bootstrap a nextvalue
        nextValue = policyReturn.valueEstimate
        # Compute targets (consisting of discounted rewards + bootstrapped value)
        rolloutstorage.computeTargets(nextValue, gamma)

        # Compute value and action losses:
        values = torch.stack(tuple(trackedValues['values']), dim=0)

        actionLogProbs = torch.stack(tuple(trackedValues['actionLogProbs']), dim=0)

        advantages = rolloutstorage.returns[:-1] - values
        valueLoss = advantages.pow(2).mean()
        actionLoss = -(Variable(advantages.detach()) * actionLogProbs).mean()

        # compute encoding loss and entropy loss
        avgEncodingLoss = torch.stack(tuple(trackedValues['totalEncodingLoss'])).mean()
        distEntropy = torch.stack(tuple(trackedValues['distEntropy'])).mean()

        totalLoss = (valueLoss * valueLossCoef
                     + actionLoss * actionLossCoef
                     - distEntropy * entropyCoef
                     + avgEncodingLoss * encodingLossCoef)

        self.optimizer.zero_grad()
        totalLoss.backward(retain_graph=retainGraph)
        self.optimizer.step()

        if not retainGraph:
            currentMemory['states'] = currentMemory['states'].detach()

        return rolloutstorage, currentMemory
