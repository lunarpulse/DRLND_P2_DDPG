import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic, OUNoise, ReplayBuffer
from prioritized_memory import Memory

BUFFER_SIZE = int(5e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # coefficient for soft update of target
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY_ACTOR = 0.0        # L2 weight decay of ACTOR
WEIGHT_DECAY_CRITIC = 0.0        # L2 weight decayof CRITIC
ONU_THETA = 0.15 # ONU noise init parameter theta
ONU_SIGMA = 0.20 # ONU noise init parameter sigma
EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 30        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay

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
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY_ACTOR)
        
        # Construct Critic networks 
        self.critic_local = Critic(self.state_size, self.action_size, self.seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY_CRITIC)
        
        self.eps = EPS_START
        self.eps_decay = 1/(EPS_EP_END)  # set decay rate based on epsilon end target
        # noise processing
        self.noise = OUNoise( action_size, action_size, random_seed, theta=ONU_THETA, sigma=ONU_SIGMA)
        
        # Replay memory
        self.memory = Memory(BUFFER_SIZE)
        
    def act(self, state, add_noise=True, eps = 1.0):
        """Returns actions for given state as per current policy."""
        # convert state from numpy to pytorch array 
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += eps * self.noise.sample()
        
        return np.clip(action, -1, 1)
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(state.shape[0]):
            self.memory.add(reward[i], (state[i, :], action[i], reward[i], next_state[i, :], done[i]))
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            mini_batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
            self.learn(mini_batch, idxs, is_weights, GAMMA)
        
    def reset(self):
        """ reset noise """
        self.noise.reset()
        
    def learn(self, experience_batch, idxs, is_weights, gamma):
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
        # states, actions, rewards, next_states, dones = experience_batch
        mini_batch = [*zip(*experience_batch)] #https://stackoverflow.com/questions/4937491/matrix-transpose-in-python

        states = torch.FloatTensor(np.vstack(mini_batch[0])).to(device)
        actions = torch.FloatTensor(list(mini_batch[1])).to(device)
        rewards = np.array(mini_batch[2])
        next_states = torch.FloatTensor(np.vstack(mini_batch[3])).to(device)
        dones = np.array(mini_batch[4])

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_next_np = Q_targets_next.detach().cpu().numpy() #(BATCH_SIZE,1)
        Q_next_np_tp = Q_next_np.transpose()
        # Compute Q targets for current states (y_i)

        dones_flipped = np.array([1 - x for x in dones])
        # Q_targets = rewards[:,ai] + (gamma * Q_targets_next * (1 - dones[:,ai]))
        mult_v = np.multiply(Q_next_np_tp, dones_flipped)
        Q_targets_np_tp = rewards + (gamma * mult_v)
        Q_targets = torch.FloatTensor(Q_targets_np_tp.transpose()).to(device)
        # Q_targets = rewards + (gamma * Q_targets_next.cpu().numpy() * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = (torch.FloatTensor(is_weights).to(device) * F.mse_loss(Q_expected, Q_targets)).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Priorities
        errors = abs(Q_expected.detach().cpu().numpy() - Q_targets_np_tp.transpose())
        # update priority
        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)   

        # update noise decay parameter
        if self.eps >= EPS_FINAL:
            self.eps -= self.eps_decay
            self.eps = max(self.eps, EPS_FINAL)
        self.noise.reset()

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