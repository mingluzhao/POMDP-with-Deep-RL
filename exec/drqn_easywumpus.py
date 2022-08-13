import matplotlib.pyplot as plt

import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch
import torch.nn.functional as f

from env.wrapper_easywumpus import EasyWumpusEnv

class RnnNet(nn.Module):
    def __init__(self,observation_dim,action_dim,hidden_dim):
        super(RnnNet,self).__init__()
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
    
    
class ReplayBuffer(object):
    def __init__(self,buffer_size,episode_maxstep,observation_dim):
        self.episode_maxstep = episode_maxstep 
        self.current_idx = 0
        self.current_size =0
        self.max_size = buffer_size
        self.observation_dim = observation_dim
        # No padded, No o_next
        # if o==0, padded
        # O_next
        self.memory = {"o": np.empty([self.max_size, self.episode_maxstep,self.observation_dim]),
                       "u": np.empty([self.max_size, self.episode_maxstep,1]),
                       "r": np.empty([self.max_size, self.episode_maxstep,1]),
                       "o_next":np.empty([self.max_size, self.episode_maxstep,self.observation_dim]),
                       # we will pad the episode so that every episode has the same length
                       # the padded timesteps will be marked as 1
                       "padded":np.empty([self.max_size, self.episode_maxstep,1]),
                       "terminate": np.empty([self.max_size, self.episode_maxstep,1])}
    
    # each time, add an entire episode to the buffer
    def addexperience(self,one_episode):
        # determine index
   
        self.memory['o'][self.current_idx] = one_episode['o']
        self.memory['u'][self.current_idx] = one_episode['u']
        self.memory['r'][self.current_idx] = one_episode['r']
        self.memory['o_next'][self.current_idx] = one_episode['o_next']
        self.memory['padded'][self.current_idx] = one_episode['padded']
        self.memory['terminate'][self.current_idx] = one_episode['terminate']
        
        
    # determine index
        self.current_size = min(self.current_size+1,self.max_size)    
        self.current_idx = 0 if self.current_idx==self.max_size-1 else self.current_idx+1
     
    
    def samplefromemory(self,minibatch_size):
        temp_buffer = {}
        if self.current_size >= minibatch_size:
            idx = random.sample(range(self.current_size), minibatch_size)
            for key in self.memory.keys():
                temp_buffer[key] = self.memory[key][idx]
          
            return temp_buffer
        else:
            return None

# e_Decay()
# init emin edecay einitial
# decay()

    
        

class DRQNAgent(object):
    def __init__(self,model,target_model,memory,gamma,minibatch_size,e, e_min,e_decay,update_freq):

        self.model = model
        self.target_model = target_model
    
        # Initialize Hidden States
        # Hidden state changes as we play
        self.hidden_state = None
        # Target_hidden doesn't change as we play, only changes when training
        self.target_hidden_state = None
        
        #Initialize the training parameters
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Initiialize the Memory
        self.memory = memory
        
        # Initialize egreedy parameters
        
        # put it together
        self.e = e
        self.e_min = e_min
        self.e_decay = e_decay
        self.minibatch_size = minibatch_size
        
        # Initialize train information
        self.train_step = 0
        self.update_freq = update_freq
        
    def init_hidden_state(self, minibatch_size):
        # when train, initialize number = minibatch_size
        # when act, initialize number = 1
        self.hidden_state = torch.zeros((minibatch_size, 1,self.model.hidden_dim))
        self.target_hidden_state = torch.zeros((minibatch_size, 1,self.target_model.hidden_dim))
        
    
    def choose_action(self,obs):
        
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_value, self.hidden_state = self.model(obs,self.hidden_state)
        if np.random.rand()<=self.e:
            return random.randrange(len(q_value[0])) 
        else:
            return torch.argmax(q_value[0]).item()
        
    def get_inputs(self, batch,transition_idx):
        # Get same transition_idx data from all sampled episodes
        
        # use timestep +1
        # o  [1,2,3,4,5,6,7,0,0,0,0,0,0,0,0,0,0]   try this
        obs, obs_next = batch['o'][:, transition_idx,:], batch['o_next'][:, transition_idx,:]
        inputs = torch.tensor(obs, dtype=torch.float32)
        inputs_next = torch.tensor(obs_next, dtype=torch.float32)
        return inputs, inputs_next
        
        
    def get_q_targetq_frombatch(self,batch,batch_max_step):
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
    
    def _get_batch_max_step(self,batch):
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
        minibatch = self.memory.samplefromemory(self.minibatch_size)
        
        if minibatch is not None:

            #cut function
            batch_max_step = self._get_batch_max_step(minibatch)
            
            for key in minibatch.keys():
                # For every episode in the minibatch, get up to batch_max_step experience
                minibatch[key] = minibatch[key][:, :batch_max_step]
            # cut

            # Initialize learning parameters function
            self.e = max(self.e_min, self.e*self.e_decay)
            self.init_hidden_state(self.minibatch_size)
            
   

            # creatTensorfrommemory function
            for key in minibatch.keys():
                if key == 'u':
                    minibatch[key] = torch.tensor(minibatch[key], dtype=torch.long)
                else:
                    minibatch[key] = torch.tensor(minibatch[key], dtype=torch.float32)


                    
            # calculateLossfrombatch function (take minibatch, return loss)
            u, r,terminal = minibatch['u'], minibatch['r'],  minibatch['terminate']
            # padded experience will have mask = 0, so laster their loss will be 0
            
            # createMask function
            # np.array(0 if obs != 0, 1 otherwise)
            mask = 1 - minibatch["padded"].float()
            
            q_preds, q_targets = self.get_q_targetq_frombatch(minibatch,batch_max_step)
            
            # q_preds is selected by action. only the actioned q_pred is modified in q_target. The non-actioned doesn't affect loss
            q_preds = (torch.gather(q_preds, dim=2, index=u)).squeeze(1)
            
            q_targets = (q_targets.max(dim=2)[0]).unsqueeze(2)
            
            targets = r+self.gamma*q_targets*(1-terminal)
            
            td_error = (q_preds - targets.detach())
            
            masked_td_error = mask * td_error
            # MSE loss 
            loss = (masked_td_error ** 2).sum() / mask.sum()




            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_step+=1
            if self.train_step%self.update_freq:
                self.target_model.load_state_dict(self.model.state_dict())
    
    
class Rollout(object):
    def __init__(self,simulator,agent,observation_shape):
        self.simulator = simulator
        self.agent = agent
        self.observation_shape = observation_shape
    
    def generate_episode(self):
        # change observation to control POMDP 
        # [::2]
        
        o,o_next, u, r,terminate, padded =[], [], [], [], [], []
        terminal = False
        step =0
        self.agent.init_hidden_state(1)
        self.reward = 0
        state = self.simulator.getInitialState()
        observation = [self.simulator.observation(state,0)]

        while not terminal and step<self.agent.memory.episode_maxstep:

           # if step <= 6:
            #    action = random.randint(0,1)
            #else:            
            action = self.agent.choose_action(observation)
       
            next_state = self.simulator.transition(state,action)
         
            next_observation = [self.simulator.observation(next_state,action)]
            reward = self.simulator.reward(state,action,next_state)
 
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
            o_next.append(np.zeros((self.observation_shape )))
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
    action_dim = 8
    
    drqn_net = RnnNet(observation_dim,action_dim,hidden_dim)
    drqn_target_net = RnnNet(observation_dim,action_dim,hidden_dim)
    drqn_target_net.load_state_dict(drqn_net.state_dict())
    
    
    buffer_size = 10000
    episode_maxstep = 20
    replay_buffer = ReplayBuffer(buffer_size,episode_maxstep,observation_dim)
    
    
    gamma = 0.9
    minibatch_size = 2
    e = 1.0
    e_min = 0.001
    e_decay = 0.999
    update_freq = 100    
    drqn_agent = DRQNAgent(drqn_net,drqn_target_net,replay_buffer,gamma,minibatch_size,e,e_min,e_decay,update_freq)
    
    
    simulator = EasyWumpusEnv(0.8)
    rollout = Rollout(simulator, drqn_agent, observation_dim)
    
    max_train_episodes = 10000
    current_episode = 0
    collect_n_episode = 1
    moving_average =[]
    scores = deque(maxlen=100)
    while current_episode<=max_train_episodes:
        print("episode {}/{}".format(current_episode,max_train_episodes))
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

        
        
            
        
    
    
    
    
    
            
            
            
        
        
        
        
        
    
        
    
