import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic, OUNoise, ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG_agent():
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """init the agent"""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        
        # Construct Actor networks
        self.actor_local = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # Construct Critic networks 
        self.critic_local = Critic(self.state_size, self.action_size, self.seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # noise processing
        self.noise = OUNoise((num_agents,action_size), random_seed)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        #self.memory = MemoryBuffer(BUFFER_SIZE, BATCH_SIZE)
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        # convert state from numpy to pytorch array 
        state = torch.from_numpy(state).float().to(device)
        # use actor_local to predict action
        # turn nn into evaluation mode 
        self.actor_local.eval()
        # turn off computing gradients
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        # turn the nn into training mode
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        
        # clipping the action from min to max
        return np.clip(action, -1, 1)
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(state.shape[0]):
            self.memory.add(state[i, :], action[i], reward[i], next_state[i, :], done[i])
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
        
    def reset(self):
        """ reset noise """
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        # mean squared error loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        # zero_grad because we do not want to accumulate 
        # gradients from other batches, so needs to be cleared
        self.critic_optimizer.zero_grad()
        # compute derivatives for all variables that
        # requires_grad-True
        critic_loss.backward()
        # update those variables that requires_grad-True
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # take the current states and predict actions
        actions_pred = self.actor_local(states)
        # -1 * (maximize) Q value for the current prediction
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        # zero_grad because we do not want to accumulate 
        # gradients from other batches, so needs to be cleared
        self.actor_optimizer.zero_grad()
        # compute derivatives for all variables that
        # requires_grad-True
        actor_loss.backward()
        # update those variables that requires_grad-True
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)