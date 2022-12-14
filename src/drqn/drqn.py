import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as f


class RnnNet(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim):
        super(RnnNet, self).__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.hidden_dim = hidden_dim

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.gru(x, h_in)
        q = self.fc2(h)
        return q, h


class ReplayBufferRnn(object):
    def __init__(self, buffer_size, episode_maxstep, observation_dim):
        self.episode_maxstep = episode_maxstep
        self.current_idx = 0
        self.current_size = 0
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        # No padded, No o_next
        # if o==0, padded
        # O_next
        self.memory = {"o": np.empty([self.buffer_size, self.episode_maxstep, self.observation_dim]),
                       "u": np.empty([self.buffer_size, self.episode_maxstep, 1]),
                       "r": np.empty([self.buffer_size, self.episode_maxstep, 1]),
                       "o_next": np.empty([self.buffer_size, self.episode_maxstep, self.observation_dim]),
                       # we will pad the episode so that every episode has the same length
                       # the padded timesteps will be marked as 1
                       "padded": np.empty([self.buffer_size, self.episode_maxstep, 1]),
                       "terminate": np.empty([self.buffer_size, self.episode_maxstep, 1])}

    # each time, add an entire episode to the buffer
    def add(self, one_episode):
        # determine index
        self.memory['o'][self.current_idx] = one_episode['o']
        self.memory['u'][self.current_idx] = one_episode['u']
        self.memory['r'][self.current_idx] = one_episode['r']
        self.memory['o_next'][self.current_idx] = one_episode['o_next']
        self.memory['padded'][self.current_idx] = one_episode['padded']
        self.memory['terminate'][self.current_idx] = one_episode['terminate']

        # determine index
        self.current_size = min(self.current_size + 1, self.buffer_size)
        self.current_idx = 0 if self.current_idx == self.buffer_size - 1 else self.current_idx + 1

    def sample(self, minibatch_size):
        temp_buffer = {}
        if self.current_size < minibatch_size:
            return None

        idx = random.sample(range(self.current_size), minibatch_size)
        for key in self.memory.keys():
            temp_buffer[key] = self.memory[key][idx]

        return temp_buffer


# e_Decay()
# init emin edecay einitial
# decay()


class DRQNAgent(object):
    def __init__(self, model, target_model, memory, gamma, minibatch_size, e, e_min, e_decay, update_freq, lr = 1e-3):
        self.model = model
        self.target_model = target_model

        # Hidden state changes as we play
        self.hidden_state = None
        # Target_hidden doesn't change as we play, only changes when training
        self.target_hidden_state = None

        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = memory

        self.e = e
        self.e_min = e_min
        self.e_decay = e_decay
        self.minibatch_size = minibatch_size

        self.train_step = 0
        self.update_freq = update_freq

    def init_hidden_state(self, minibatch_size):
        # when train, initialize number = minibatch_size
        # when act, initialize number = 1
        self.hidden_state = torch.zeros((minibatch_size, 1, self.model.hidden_dim))
        self.target_hidden_state = torch.zeros((minibatch_size, 1, self.target_model.hidden_dim))

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_value, self.hidden_state = self.model(obs, self.hidden_state)
        if np.random.rand() <= self.e:
            return random.randrange(len(q_value[0]))
        else:
            return torch.argmax(q_value[0]).item()

    def get_inputs(self, batch, transition_idx):
        # Get same transition_idx data from all sampled episodes

        # use timestep +1
        # o  [1,2,3,4,5,6,7,0,0,0,0,0,0,0,0,0,0]   try this
        obs, obs_next = batch['o'][:, transition_idx, :], batch['o_next'][:, transition_idx, :]
        inputs = torch.tensor(obs, dtype=torch.float32)
        inputs_next = torch.tensor(obs_next, dtype=torch.float32)
        return inputs, inputs_next

    def get_q_targetq_frombatch(self, batch, batch_max_step):
        q_preds, q_targets = [], []
        # initialize hidden
        for timestep in range(batch_max_step):
            inputs, inputs_next = self.get_inputs(batch, timestep)
            q_pred, self.hidden_state = self.model(inputs, self.hidden_state)
            q_target, self.target_hidden_state = self.target_model(inputs_next, self.target_hidden_state)
            q_preds.append(q_pred)
            q_targets.append(q_target)
        # (batch_size, batch_max_step, 2)
        q_preds = torch.stack(q_preds, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_preds, q_targets

    def _get_batch_max_step(self, batch):
        terminated = batch['terminate']
        # observstion_history
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.memory.episode_maxstep):
                if terminated[episode_idx, transition_idx] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:
            max_episode_len = self.memory.episode_maxstep
        return max_episode_len

    def learn(self):
        minibatch = self.memory.sample(self.minibatch_size)

        if minibatch is not None:
            # cut function
            batch_max_step = self._get_batch_max_step(minibatch)

            for key in minibatch.keys():
                minibatch[key] = minibatch[key][:, :batch_max_step]
            # cut

            # Initialize learning parameters function
            self.e = max(self.e_min, self.e * self.e_decay)
            self.init_hidden_state(self.minibatch_size)

            # creatTensorfrommemory function
            for key in minibatch.keys():
                if key == 'u':
                    minibatch[key] = torch.tensor(minibatch[key], dtype=torch.long)
                else:
                    minibatch[key] = torch.tensor(minibatch[key], dtype=torch.float32)

            # calculateLossfrombatch function (take minibatch, return loss)
            u, r, terminal = minibatch['u'], minibatch['r'], minibatch['terminate']
            # padded experience will have mask = 0, so later their loss will be 0

            # createMask function
            # np.array(0 if obs != 0, 1 otherwise)
            mask = 1 - minibatch["padded"].float()

            q_preds, q_targets = self.get_q_targetq_frombatch(minibatch, batch_max_step)

            # q_preds is selected by action. only the actioned q_pred is modified in q_target. The non-actioned doesn't affect loss
            q_preds = (torch.gather(q_preds, dim=2, index=u)).squeeze(1)
            q_targets = (q_targets.max(dim=2)[0]).unsqueeze(2)

            targets = r + self.gamma * q_targets * (1 - terminal)
            td_error = (q_preds - targets.detach())

            masked_td_error = mask * td_error
            # MSE loss
            loss = (masked_td_error ** 2).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_step += 1
            if self.train_step % self.update_freq:
                self.target_model.load_state_dict(self.model.state_dict())
