import random
import gym
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch as T
import matplotlib.pyplot as plt

from wrapper_tiger import TigerEnv



class BuildModel(nn.Module):
    def __init__(self, lr, layers,input_dimension):
        super(BuildModel, self).__init__()
        self.layers=nn.ModuleList(layers)
        self.learning_rate = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.input_dimension=input_dimension

    def forward(self, state):
        # This model needs to reshape the input from simulator before calculating
        state = np.reshape(state, [1, self.input_dimension])
        state = T.tensor(state, dtype=T.float32).to(self.device)
        for layer in self.layers:
            state = layer(state)
        return state

# Tested     
def policyEgreedy(Q,e):
    if np.random.rand()<=e:
        return random.randrange(len(Q[0])) 
    else:
        return T.argmax(Q[0]).item()


# Tested
def sampleFromMemory(minibatchSize,memory):
    if len(memory)<minibatchSize:
        return []
    else:
        sample = random.sample(memory,minibatchSize)  
        return sample

# Tested
class LearnFromOneSample(object):
    def __init__(self, gamma):
        self.gamma=gamma
        
    def __call__(self, model, target_model,sample):
        state, action, reward, next_state,terminal =sample
        pred = model(state)
        # test yixa
        target=reward+self.gamma*target_model(next_state)[0].max() if not terminal else reward
        target_f=pred.clone()
        target_f[0][action]=target
        return pred[0], target_f[0]
    

# Tested 
def learnbackprop(model,target_model,minibatch,learnFromOneSample):
    random.shuffle(minibatch)
    model.optimizer.zero_grad()
    pred_target_pair = [learnFromOneSample(model, target_model,episode)for episode in minibatch]
    # the first list unfolds the zip iterator.
    # the second list convert tuple into list
    pred_batch = T.stack(list(list(zip(*pred_target_pair))[0]))
    target_batch = T.stack(list(list(zip(*pred_target_pair))[1]))
    loss = model.loss(target_batch, pred_batch).to(model.device)
    loss.backward()
    model.optimizer.step()
    return model        


# Tested
class LearnFromMemory(object):
    def __init__(self, learnFromOneSample, train_freq, backprop):
        self.learnFromOneSample=learnFromOneSample
        self.train_freq = train_freq
        self.backprop = backprop
        
    def __call__(self, model, target_model,minibatch):
        if minibatch!=[] and np.random.random()<self.train_freq:
            # only train train_freq (default is 0.25) of the times
            model = self.backprop(model,target_model,minibatch,self.learnFromOneSample)
            return model,target_model
        else:
            # don't train if minibatch is empty and we don't train it 75% the steps
            return model, target_model

    
# tested
class SimulateOneStep(object):
    def __init__(self,transition,reward,observation, isterminal):
        self.transition = transition
        self.reward = reward
        self.observation = observation
        self.isterminal = isterminal

    def __call__(self,state,action):

        next_observation = self.observation(state,action)
        next_state = self.transition(state,action) 
        reward = self.reward(state,action,next_state)
        terminal = self.isterminal(next_state)
        
        return reward,next_state,next_observation,terminal


# tested
class TrainOneStep(object): 
    def __init__(self, policyEgreedy,simulateOneStep,sampleFromMemory,learnFromMemory,minibatchSize): 
        self.policyEgreedy=policyEgreedy 
        self.simulateOneStep = simulateOneStep
        self.sampleFromMemory=sampleFromMemory
        self.learnFromMemory=learnFromMemory
        self.minibatchSize = minibatchSize
      
        
    def __call__(self, model,target_model, memory, state,observation,e):
        
        Q=model(observation)
        action = self.policyEgreedy(Q,e)
        reward, next_state,next_observation,terminal =self.simulateOneStep(state,action)
        memory.append([observation, action,reward,next_observation,terminal])
        minibatch=self.sampleFromMemory(self.minibatchSize,memory)
        model,target_model=self.learnFromMemory(model, target_model,minibatch)
        return model,target_model, memory,next_state, next_observation,terminal,reward


# tested
class GetEpsilon(object):
    def __init__(self,e,e_min,e_decay):
        self.e =e
        self.e_min=e_min
        self.e_decay = e_decay

    def __call__(self):
        self.e = max(self.e_min, self.e*self.e_decay)
        return self.e


# tested
class Train(object):
    def __init__(self, trainOneStep, maxSteps, maxEpisodes,getEpsilon, update_freq):
        self.trainOneStep=trainOneStep
        self.maxSteps=maxSteps
        self.maxEpisodes = maxEpisodes
        self.getEpsilon = getEpsilon
        self.update_freq = update_freq

    def __call__(self, model, target_model,memory, simulator):
        training_score = deque(maxlen = 100)
        moving_average = []
        for episode in range(self.maxEpisodes):
            state = simulator.getInitialState()
            # observe nothing at first
            observation = [3]
            e = self.getEpsilon()
            total_reward = 0
            for step in range(self.maxSteps):
                model, target_model,memory,state,observation,terminal,reward =self.trainOneStep(model, target_model,memory,state,observation,e)
                total_reward += reward
                if step%self.update_freq==0:
                    target_model.load_state_dict(model.state_dict())
                if terminal:
                    print("epsiode {}/{}".format(episode,self.maxEpisodes))
                    print("current e {:.2}".format(e))
                    print("score is {}".format(total_reward))
                    training_score.append(total_reward)
                    moving_average.append(np.mean(training_score))
                    break
        return model,moving_average
        
        
def main():
    observation_dimension = 1  
    action_dimension = 3
    simulator = TigerEnv()
    layers=[nn.Linear(observation_dimension, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
            nn.Linear(24, action_dimension)]

    model=BuildModel(lr=0.001, layers=layers,input_dimension = observation_dimension)
    target_model = BuildModel(lr=0.001, layers=layers,input_dimension = observation_dimension)
    target_model.load_state_dict(model.state_dict())

    memory=deque(maxlen=10000)
    minibatchSize=32
    gamma=0.95
    train_freq = 0.25
    learnFromOneSample=LearnFromOneSample(gamma)
    learnFromMemory=LearnFromMemory(learnFromOneSample,train_freq,learnbackprop)
    simulateOneStep = SimulateOneStep(simulator.transition,simulator.reward,simulator.observation,simulator.isterminal)
    trainOneStep=TrainOneStep(policyEgreedy,simulateOneStep,sampleFromMemory,learnFromMemory,minibatchSize)
    
    
    e=1.0
    decay_rate = 0.999
    e_min=0.01
    getEpsilon = GetEpsilon(e,e_min,decay_rate)

    
    maxSteps=20
    maxEpisodes=10000
    target_model_update_freq = 100
    train=Train(trainOneStep, maxSteps,maxEpisodes,getEpsilon,target_model_update_freq)
    model,scores=train(model, target_model,memory, simulator)

    

    plt.plot(scores)
    plt.show()

    

if __name__ == '__main__':
    main()

