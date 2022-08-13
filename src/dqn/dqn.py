import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from collections import deque

class BuildModel(nn.Module):
    def __init__(self, lr, layers, input_dimension):
        super(BuildModel, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.learning_rate = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.input_dimension = input_dimension

    def forward(self, state):
        # This model needs to reshape the input from simulator before calculating
        state = np.reshape(state, [1, self.input_dimension])
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        for layer in self.layers:
            state = layer(state)
        return state


# Tested
def policyEgreedy(Q, e):
    if np.random.rand() <= e:
        return random.randrange(len(Q[0]))
    else:
        return torch.argmax(Q[0]).item()


# Tested
def sampleFromMemory(minibatchSize, memory):
    if len(memory) < minibatchSize:
        return []
    else:
        sample = random.sample(memory, minibatchSize)
        return sample


# Tested
class LearnFromOneSample(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, model, target_model, sample):
        state, action, reward, next_state, terminal = sample
        pred = model(state)
        # unittest yixa
        target = reward + self.gamma * target_model(next_state)[0].max() if not terminal else reward
        target_f = pred.clone()
        target_f[0][action] = target
        return pred[0], target_f[0]


# Tested
def learnbackprop(model, target_model, minibatch, learnFromOneSample):
    random.shuffle(minibatch)
    model.optimizer.zero_grad()
    pred_target_pair = [learnFromOneSample(model, target_model, episode) for episode in minibatch]
    # the first list unfolds the zip iterator.
    # the second list convert tuple into list
    pred_batch = torch.stack(list(list(zip(*pred_target_pair))[0]))
    target_batch = torch.stack(list(list(zip(*pred_target_pair))[1]))
    loss = model.loss(target_batch, pred_batch).to(model.device)
    loss.backward()
    model.optimizer.step()
    return model


# Tested
class LearnFromMemory(object):
    def __init__(self, learnFromOneSample, train_freq, backprop):
        self.learnFromOneSample = learnFromOneSample
        self.train_freq = train_freq
        self.backprop = backprop

    def __call__(self, model, target_model, minibatch):
        if minibatch != [] and np.random.random() < self.train_freq:
            # only train train_freq (default is 0.25) of the times
            model = self.backprop(model, target_model, minibatch, self.learnFromOneSample)
            return model, target_model
        else:
            # don't train if minibatch is empty and we don't train it 75% the steps
            return model, target_model


# tested
class SimulateOneStep(object):
    def __init__(self, transition, reward, observation, isterminal):
        self.transition = transition
        self.reward = reward
        self.observation = observation
        self.isterminal = isterminal

    def __call__(self, state, action):
        next_observation = self.observation(state, action)
        next_state = self.transition(state, action)
        reward = self.reward(state, action, next_state)
        terminal = self.isterminal(next_state)

        return reward, next_state, next_observation, terminal


# tested
class TrainOneStep(object):
    def __init__(self, policyEgreedy, simulateOneStep, sampleFromMemory, learnFromMemory, minibatchSize):
        self.policyEgreedy = policyEgreedy
        self.simulateOneStep = simulateOneStep
        self.sampleFromMemory = sampleFromMemory
        self.learnFromMemory = learnFromMemory
        self.minibatchSize = minibatchSize

    def __call__(self, model, target_model, memory, state, observation, e):
        Q = model(observation)
        action = self.policyEgreedy(Q, e)
        reward, next_state, next_observation, terminal = self.simulateOneStep(state, action)
        memory.append([observation, action, reward, next_observation, terminal])
        minibatch = self.sampleFromMemory(self.minibatchSize, memory)
        model, target_model = self.learnFromMemory(model, target_model, minibatch)
        return model, target_model, memory, next_state, next_observation, terminal, reward


# tested
class GetEpsilon(object):
    def __init__(self, e, e_min, e_decay):
        self.e = e
        self.e_min = e_min
        self.e_decay = e_decay

    def __call__(self):
        self.e = max(self.e_min, self.e * self.e_decay)
        return self.e
