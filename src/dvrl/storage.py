import torch


class RolloutStorage(object):
    def __init__(self, num_steps):
        # def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.masks = torch.ones(num_steps + 1, 1, 1)
        self.rewards = torch.zeros(num_steps + 1, 1, 1)

        # Computed later
        self.returns = torch.zeros(num_steps + 1, 1, 1)

    def insert(self, step, reward, mask):
        self.masks[step + 1].copy_(mask)
        self.rewards[step + 1].copy_(reward)

    def computeTargets(self, next_value, gamma):
        # fill bootstrapped value here
        self.returns[-1] = next_value

        for step in reversed(range(self.rewards.size(0) - 1)):
            self.returns[step] = self.returns[step + 1] * \
                                 gamma * self.masks[step + 1] + self.rewards[step + 1]
