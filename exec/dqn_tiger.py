import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from collections import deque
from src.dqn.dqn import BuildModel, policyEgreedy, sampleFromMemory, LearnFromOneSample, learnbackprop, \
    LearnFromMemory, SimulateOneStep, TrainOneStep, GetEpsilon
from env.wrapper_tiger import TigerEnv

# tested
class Train(object):
    def __init__(self, trainOneStep, maxSteps, maxEpisodes, getEpsilon, update_freq):
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
                model, target_model, memory, state, observation, terminal, reward = self.trainOneStep(model, target_model, memory, state, observation, e)
                total_reward += reward
                if step % self.update_freq == 0:
                    target_model.load_state_dict(model.state_dict())
                if terminal:
                    print("epsiode {}/{}".format(episode,self.maxEpisodes))
                    print("current e {:.2}".format(e))
                    print("score is {}".format(total_reward))
                    training_score.append(total_reward)
                    moving_average.append(np.mean(training_score))
                    break
        return model, moving_average
        
        
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

