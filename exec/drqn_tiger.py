from env.wrapper_tiger import TigerEnv
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

from src.drqn.drqn import RnnNet, ReplayBuffer, DRQNAgent


class Rollout(object):
    def __init__(self, simulator, agent, observation_shape):
        self.simulator = simulator
        self.agent = agent
        self.observation_shape = observation_shape

    def generate_episode(self):
        # change observation to control POMDP 
        # [::2]

        o, o_next, u, r, terminate, padded = [], [], [], [], [], []
        terminal = False
        step = 0
        self.agent.init_hidden_state(1)
        self.reward = 0
        state = self.simulator.getInitialState()
        observation = [2]

        while not terminal and step < self.agent.memory.episode_maxstep:

            if step <= 4:
                action = 2
            else:
                action = self.agent.choose_action(observation)

            next_state = self.simulator.transition(state, action)

            next_observation = [self.simulator.observation(state, action)]
            reward = self.simulator.reward(state, action, next_state)

            self.reward += reward
            terminal = self.simulator.isterminal(next_state)

            o.append(observation)
            o_next.append(next_observation)
            u.append([action])
            r.append([reward])
            terminate.append([1.0 if terminal else 0.0])
            padded.append([0.0])
            step += 1
            observation = next_observation
            state = next_state

        # padding
        # move to train
        for i in range(step, self.agent.memory.episode_maxstep):
            o.append(np.zeros((self.observation_shape)))
            u.append(np.zeros((1)))
            r.append([0.0])
            o_next.append(np.zeros((self.observation_shape)))
            padded.append([1.0])
            terminate.append([1.0])

        episode = dict(o=o.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       padded=padded.copy(),
                       terminate=terminate.copy()
                       )

        print("score :{}".format(self.reward))
        return episode, self.reward


def main():
    observation_dim = 1
    hidden_dim = 32
    action_dim = 3

    drqn_net = RnnNet(observation_dim, action_dim, hidden_dim)
    drqn_target_net = RnnNet(observation_dim, action_dim, hidden_dim)
    drqn_target_net.load_state_dict(drqn_net.state_dict())

    buffer_size = 10000
    episode_maxstep = 20
    replay_buffer = ReplayBuffer(buffer_size, episode_maxstep, observation_dim)

    gamma = 0.9
    minibatch_size = 2
    e = 1.0
    e_min = 0.01
    e_decay = 0.999
    update_freq = 100
    drqn_agent = DRQNAgent(drqn_net, drqn_target_net, replay_buffer, gamma, minibatch_size, e, e_min, e_decay,
                           update_freq)

    simulator = TigerEnv()
    rollout = Rollout(simulator, drqn_agent, observation_dim)

    max_train_episodes = 10000
    current_episode = 0
    collect_n_episode = 1

    moving_average = []
    scores = deque(maxlen=100)
    total_scores = []
    while current_episode <= max_train_episodes:
        print("episode {}/{}".format(current_episode, max_train_episodes))
        print("epsilon {:.2}".format(drqn_agent.e))
        episode, reward = rollout.generate_episode()
        scores.append(reward)

        moving_average.append(np.mean(scores))
        drqn_agent.memory.addexperience(episode)
        drqn_agent.learn()
        current_episode += 1

    plt.plot(moving_average)

    plt.show()


if __name__ == '__main__':
    main()
