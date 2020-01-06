import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ProximalPolicyOptimisation:
    buffer_attrs = [
        "states", "actions", "next_states",
        "rewards", "log_probs", "values", "dones",
    ]

    def __init__(self, env, state_dim, action_dim, hiddens=[256, 128],  tmax=128, n_epoch=10, batch_size=128,
                 gamma=0.99, gae_lambda=0.96, eps=0.10, device="cpu"):
        self.env = env
        self.model = GaussianActorCriticNetwork(state_dim, action_dim, hiddens).to(device)
        self.opt_model = optim.Adam(self.model.parameters(), lr=1e-4)
        self.state_dim = self.model.state_dim
        self.action_dim = self.model.action_dim
        self.tmax = tmax
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps = eps
        self.device = device

        self.rewards = None
        self.scores_by_episode = []

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.target_q_value = []

        self.GAE_log_prob = []
        self.GAE_entropy = []
        
        self.reset()

    def to_tensor(self, x, dtype=np.float32):
        return torch.from_numpy(np.array(x).astype(dtype)).to(self.device)

    def reset(self):
        ''' clear all the buffer lists '''
        self.batch_number = 0
        self.actions.clear()
        self.rewards.clear()
        self.states.clear()
        self.next_states.clear()
        self.dones.clear()
        self.target_q_value.clear()
        self.GAE_log_prob.clear()
        self.GAE_entropy.clear()

    def act(self, states):
        self.model.eval()
        with torch.no_grad():
            actions, log_prob, entropy, target_q_value = self.model.actor_act(states)
        self.model.train()
        self.GAE_log_prob.append(log_prob.unsqueeze(0))
        self.GAE_entropy.append(entropy.unsqueeze(0))
        self.states.append(states.unsqueeze(0))
        self.actions.append(actions.unsqueeze(0)) # all torch tensor
        self.target_q_value.append(target_q_value.unsqueeze(0))

        return actions
        
    def calc_returns(self, rewards, values, dones, last_values):
        n_step, n_agent = rewards.shape

        # Create empty buffer
        GAE = torch.zeros_like(rewards).float().to(self.device)
        returns = torch.zeros_like(rewards).float().to(self.device)

        # Set start values
        GAE_current = torch.zeros(n_agent).float().to(self.device)
        returns_current = last_values
        values_next = last_values

        for irow in reversed(range(n_step)):
            values_current = values[irow]
            rewards_current = rewards[irow]
            gamma = self.gamma * (1. - dones[irow].float())

            # Calculate TD Error
            td_error = rewards_current + gamma * values_next - values_current
            # Update GAE, returns
            GAE_current = td_error + gamma * self.gae_lambda * GAE_current
            returns_current = rewards_current + gamma * returns_current
            # Set GAE, returns to buffer
            GAE[irow] = GAE_current
            returns[irow] = returns_current

            values_next = values_current

        return GAE, returns

    def step(self, states, actions, rewards, next_states, dones):
        '''Save the env_infor to the agent's storage, when reached to the epoch, process all synchronously'''
        # Collect Trajetories
        self.rewards.append(rewards)
        self.next_states.append(next_states)
        self.dones.append(dones)

        # here to be passed after iteration batch 
        if self.batch_number == self.batch_size - 1 :
            self.batch_number = 0
            # self.model.eval()
            # Calculate Score (averaged over agents)
            # score = torch.cat(self.rewards, dim=0).sum(dim=0).mean()

            # Append Values collesponding to last states
            # Get the expected_value
            Expected_Q_values = self.model.critic_expect(next_states).detach()
            advantages, returns = self.calc_returns(torch.stack(self.rewards),
                                                    torch.cat(self.target_q_value, dim=0),
                                                    torch.stack(self.dones),
                                                    Expected_Q_values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            # concat all info
            actions = torch.stack(self.actions)
            actions = actions.reshape([-1, actions.shape[-1]])
            states = torch.stack(self.states)
            states = states.reshape([-1, states.shape[-1]])
            next_states = torch.stack(self.next_states)
            next_states = next_states.reshape([-1, next_states.shape[-1]])
            rewards = torch.stack(self.rewards).reshape([-1])
            dones = torch.stack(self.dones).reshape([-1])
            # target_q_value = torch.cat(self.target_q_value, dim=0).reshape([-1])
            log_probs = torch.stack(self.GAE_log_prob).reshape([-1])

            advantages = advantages.reshape([-1])
            returns = returns.reshape([-1])

            # Mini-batch update
            self.model.train()
            n_sample = advantages.shape[0]
            idx = np.arange(n_sample)
            np.random.shuffle(idx)
            #shuffle all
            states_mixed, actions_mixed, log_probs_mixed = states[idx], actions[idx], log_probs[idx]
            advantages, returns = advantages[idx], returns[idx]
            
            n_batch = (n_sample - 1) // self.batch_size + 1
            for i_epoch in range(self.n_epoch):
                for i_batch in range(n_batch):
                    idx_start = self.batch_size * i_batch
                    idx_end = self.batch_size * (i_batch + 1)

                    states = states_mixed[idx_start:idx_end]
                    actions= actions_mixed[idx_start:idx_end]
                    old_log_probs = log_probs_mixed[idx_start:idx_end]
                    advantages_batch = advantages[idx_start:idx_end]
                    returns_batch = returns[idx_start:idx_end]

                    _, log_probs, entropy, values = self.model(states, actions)
                    ratio = torch.exp(log_probs - old_log_probs)
                    ratio_clamped = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                    adv_PPO = torch.min(ratio * advantages_batch, ratio_clamped * advantages_batch)
                    loss_actor = -torch.mean(adv_PPO) - 0.01 * entropy.mean()
                    loss_critic = 0.5 * (returns_batch - values).pow(2).mean()
                    loss = loss_actor + loss_critic

                    self.opt_model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
                    self.opt_model.step()
                    del(loss)
        else:
            self.batch_number += 1
        
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

class GaussianActorCriticNetwork(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hiddens=[64, 64]):
        super(GaussianActorCriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_hidden = FCNetwork(state_dim, hiddens)
        self.fc_actor = nn.Linear(hiddens[-1], action_dim)
        self.fc_critic = nn.Linear(hiddens[-1], 1)
        self.sigma = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states, actions=None):
        phi = self.fc_hidden(states)
        Q_value = self.fc_critic(phi).squeeze(-1)
        
        mu = F.tanh(self.fc_actor(phi))
        dist = torch.distributions.Normal(mu, F.softplus(self.sigma))
        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)

        return actions, log_prob, entropy, Q_value

    def actor_act(self, states):
        phi = self.fc_hidden(states)
        target_q_value = self.fc_critic(phi).squeeze(-1)
        mu = F.tanh(self.fc_actor(phi))
        dist = torch.distributions.Normal(mu, F.softplus(self.sigma))
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        return actions, log_prob, entropy, target_q_value

    def critic_expect(self, states):
        phi = self.fc_hidden(states)
        return self.fc_critic(phi).squeeze(-1)

class FCNetwork(nn.Module):
    def __init__(self, input_dim, hiddens, func=F.leaky_relu):
        super(FCNetwork, self).__init__()
        self.func =  func

        # Input Layer
        fc_first = nn.Linear(input_dim, hiddens[0])
        self.layers = nn.ModuleList([fc_first])
        # Hidden Layers
        layer_sizes = zip(hiddens[:-1], hiddens[1:])
        self.layers.extend([nn.Linear(h1, h2)
                            for h1, h2 in layer_sizes])

        def xavier(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
        self.layers.apply(xavier)

    def forward(self, x):
        for layer in self.layers:
            x = self.func(layer(x))

        return x