import random
import gym
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch as T
import matplotlib.pyplot as plt


class Wrapper(object):
    def __init__(self,env_name):
        self.env = env = gym.make(env_name)
        self.step=0
        self.current_step_reward=0
        self.current_step_termination = False

    # call this function to initialize the environment
    def get_initial_state(self):
        state_0 = self.env.reset()
        return state_0


    def transition(self,state,action):
        next_state, reward, done, _=self.env.step(action)
        self.step+=1
        self.current_step_reward = reward
        self.current_step_termination = done
        return next_state

    def reward(self,state,action,next_state):
        if self.current_step_termination:
            if self.step!= 499:
                additional_reward =-5
            elif self.step==499:
                additional_reward =5
        else:
            additional_reward = 0
        return self.current_step_reward + additional_reward

    def check_if_terminal(self,state):
        return self.current_step_termination


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
    def __init__(self,transition,reward,check_if_terminal):
        self.transition = transition
        self.reward = reward
        self.check_if_terminal = check_if_terminal

    def __call__(self,state,action):
        next_state = self.transition(state,action) 
        reward = self.reward(state,action,next_state)
        terminal = self.check_if_terminal(next_state)
        return reward,next_state,terminal


# tested
class TrainOneStep(object): 
    def __init__(self, policyEgreedy,simulateOneStep,sampleFromMemory,learnFromMemory,minibatchSize): 
        self.policyEgreedy=policyEgreedy 
        self.simulateOneStep = simulateOneStep
        self.sampleFromMemory=sampleFromMemory
        self.learnFromMemory=learnFromMemory
        self.minibatchSize = minibatchSize
      
        
    def __call__(self, model,target_model, memory, state,e): 
        Q=model(state)
        action = self.policyEgreedy(Q,e)
        reward, next_state,terminal =self.simulateOneStep(state,action)
        memory.append([state, action,reward,next_state,terminal])
        minibatch=self.sampleFromMemory(self.minibatchSize,memory)
        model,target_model=self.learnFromMemory(model, target_model,minibatch)
        return model,target_model, memory,next_state, terminal


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
        for episode in range(self.maxEpisodes):
            state = simulator.get_initial_state()
            e = self.getEpsilon()
            for step in range(self.maxSteps):
                model, target_model,memory,state,terminal =self.trainOneStep(model, target_model,memory,state,e)
                if step%self.update_freq==0:
                    target_model.load_state_dict(model.state_dict())
                if terminal:
                    break
        return model
        
        
def main():
    observation_dimension = 4  
    action_dimension = 2
    simulator = Wrapper('CartPole-v1')
    layers=[nn.Linear(observation_dimension, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
            nn.Linear(24, action_dimension)]

    model=BuildModel(lr=0.001, layers=layers,input_dimension = observation_dimension)
    target_model = BuildModel(lr=0.001, layers=layers,input_dimension = observation_dimension)
    target_model.load_state_dict(model.state_dict())

    memory=deque(maxlen=1000)
    minibatchSize=64
    gamma=0.95
    train_freq = 0.25
    learnFromOneSample=LearnFromOneSample(gamma)
    learnFromMemory=LearnFromMemory(learnFromOneSample,train_freq,learnbackprop)
    simulateOneStep = SimulateOneStep(simulator.transition,simulator.reward,simulator.check_if_terminal)
    trainOneStep=TrainOneStep(policyEgreedy,simulateOneStep,sampleFromMemory,learnFromMemory,minibatchSize)
    
    
    e=1.0
    decay_rate = 0.99
    e_min=0.01
    getEpsilon = GetEpsilon(e,e_min,decay_rate)

    
    maxSteps=500
    maxEpisodes=10000
    target_model_update_freq = 10
    train=Train(trainOneStep, maxSteps,maxEpisodes,getEpsilon,target_model_update_freq)
    model=train(model, target_model,memory, simulator)


if __name__ == '__main__':
    main()

