import numpy as np
from collections import deque
import torch.nn as nn
import matplotlib.pyplot as plt
from src.dqn.dqn import BuildModel, policyEgreedy, sampleFromMemory, LearnFromOneSample, learnbackprop, \
    LearnFromMemory, SimulateOneStep, TrainOneStep, GetEpsilon
from env.wrapper_cartpole import EnvCartpolePartial


# tested
class Train(object):
    def __init__(self, trainOneStep, maxSteps, maxEpisodes, getEpsilon, update_freq):
        self.trainOneStep = trainOneStep
        self.maxSteps = maxSteps
        self.maxEpisodes = maxEpisodes
        self.getEpsilon = getEpsilon
        self.update_freq = update_freq

    def __call__(self, model, target_model, memory, simulator):
        moving_average = []
        scores = deque(maxlen=100)
        for episode in range(self.maxEpisodes):
            state = simulator.get_initial_state()
            e = self.getEpsilon()
            for step in range(self.maxSteps):
                model, target_model, memory, state, terminal = self.trainOneStep(model, target_model, memory, state, e)
                if step % self.update_freq == 0:
                    target_model.load_state_dict(model.state_dict())
                if terminal:
                    print("epsiode {}/{}".format(episode, self.maxEpisodes))
                    print("current e {:.2}".format(e))
                    print("score is {}".format(step))
                    scores.append(step)
                    moving_average.append(np.mean(scores))
                    break
        return model, moving_average


def main():
    observation_dimension = 2
    action_dimension = 2
    simulator = EnvCartpolePartial('CartPole-v1')
    layers = [nn.Linear(observation_dimension, 24), nn.ReLU(),
              nn.Linear(24, 24), nn.ReLU(),
              nn.Linear(24, action_dimension)]

    model = BuildModel(lr=0.001, layers=layers, input_dimension=observation_dimension)
    target_model = BuildModel(lr=0.001, layers=layers, input_dimension=observation_dimension)
    target_model.load_state_dict(model.state_dict())

    memory = deque(maxlen=1000)
    minibatchSize = 64
    gamma = 0.95
    train_freq = 0.25
    learnFromOneSample = LearnFromOneSample(gamma)
    learnFromMemory = LearnFromMemory(learnFromOneSample, train_freq, learnbackprop)
    simulateOneStep = SimulateOneStep(simulator.transition, simulator.reward, simulator.check_if_terminal)
    trainOneStep = TrainOneStep(policyEgreedy, simulateOneStep, sampleFromMemory, learnFromMemory, minibatchSize)

    e = 1.0
    decay_rate = 0.99
    e_min = 0.01
    getEpsilon = GetEpsilon(e, e_min, decay_rate)

    maxSteps = 500
    maxEpisodes = 6000
    target_model_update_freq = 10
    train = Train(trainOneStep, maxSteps, maxEpisodes, getEpsilon, target_model_update_freq)
    model, moving_average = train(model, target_model, memory, simulator)

    plt.plot(moving_average)
    plt.title('DQN Solve Partial Cartpole')
    plt.ylabel('Score')
    plt.xlabel('Episodes')
    plt.show()


if __name__ == '__main__':
    main()
