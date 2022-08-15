from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from src.drqn.drqn import RnnNet, ReplayBufferRnn, DRQNAgent


class Rollout(object):
    def __init__(self, env, agent, observation_shape):
        self.env = env
        self.agent = agent
        self.observation_shape = observation_shape

    def generate_episode(self):
        # change observation to control POMDP 
        # 
        observation = self.env.reset()[::2]
        o, o_next, u, r, terminate, padded = [], [], [], [], [], []
        terminal = False
        step = 0
        self.agent.init_hidden_state(1)

        while not terminal and step < self.agent.memory.episode_maxstep:
            action = self.agent.act(observation)
            next_observation, reward, terminal, _ = self.env.step(action)
            next_observation = next_observation[::2]
            o.append(observation)
            # change observation to control POMDP
            o_next.append(next_observation)
            observation = next_observation
            u.append([action])
            r.append([reward])
            terminate.append([1.0 if terminal else 0.0])
            padded.append([0.0])
            step += 1

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

        print("score :{}".format(step))
        return episode, step


def main():
    observation_dim = 2
    hidden_dim = 32
    action_dim = 2

    drqn_net = RnnNet(observation_dim, action_dim, hidden_dim)
    drqn_target_net = RnnNet(observation_dim, action_dim, hidden_dim)
    drqn_target_net.load_state_dict(drqn_net.state_dict())

    buffer_size = 10000
    episode_maxstep = 500
    replay_buffer = ReplayBufferRnn(buffer_size, episode_maxstep, observation_dim)

    gamma = 0.9
    minibatch_size = 2
    e = 1.0
    e_min = 0.01
    e_decay = 0.99
    update_freq = 10
    drqn_agent = DRQNAgent(drqn_net, drqn_target_net, replay_buffer, gamma, minibatch_size, e, e_min, e_decay,
                           update_freq)

    env = gym.make('CartPole-v1')
    rollout = Rollout(env, drqn_agent, observation_dim)

    max_train_episodes = 60000
    current_episode = 0
    collect_n_episode = 1
    scores = deque(maxlen=100)
    moving_average = []
    while current_episode <= max_train_episodes:
        print("episode {}/{}".format(current_episode, max_train_episodes))
        print("epsilon {:.2}".format(drqn_agent.e))
        episode, step = rollout.generate_episode()
        scores.append(step)
        moving_average.append(np.mean(scores))
        drqn_agent.memory.add(episode)
        drqn_agent.learn()
        current_episode += 1

    plt.plot(moving_average)
    plt.title('DRQN Solve Cartpole')
    plt.ylabel('Score')
    plt.xlabel('Episodes')
    plt.show()


if __name__ == '__main__':
    main()
