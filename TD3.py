import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import OUNoise, ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 128)
		self.l3 = nn.Linear(128, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 128)
		self.l3 = nn.Linear(128, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 128)
		self.l6 = nn.Linear(128, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		num_agents,
		discount=0.99,
		tau=1e-3,
		batch_size=128,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		buffer_size = int(1e5),
		random_seed = 0,
		noise_theta = 0.15,
		noise_sima = 0.20,
		eps_start = 5.0,
		eps_final = 0.0,
		eps_decay = 5.0/2000
	):
		self.num_agents = num_agents;
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.batch_size = batch_size
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.memory = ReplayBuffer(buffer_size, batch_size, random_seed)
		self.noise = OUNoise( action_dim, random_seed, theta=noise_theta, sigma=noise_sima)
		self.eps = eps_start
		self.eps_final = eps_final
		self.eps_decay = eps_decay
		self.total_it = 0

	def reset(self):
		""" reset noise """
		self.noise.reset()

	def act(self, state, add_noise=True, eps = 1.0):
		state = torch.FloatTensor(state.reshape(self.num_agents, -1)).to(device)
		self.actor.eval()
		with torch.no_grad():
			action = self.actor(state).cpu().data.numpy()
		self.actor.train()
		if add_noise:
			action += self.eps * self.noise.sample()

		return np.clip(action, -1, 1).flatten().reshape(self.num_agents, -1)

	def step(self, state, action, reward, next_state, done):
		"""Save experience in replay memory, and use random sample from buffer to learn."""
		# Save experience / reward
		for i in range(state.shape[0]):
			self.memory.add(state[i, :], action[i], reward[i], next_state[i, :], done[i])

		# Learn, if enough samples are available in memory
		if len(self.memory) > self.batch_size:
			experiences = self.memory.sample()
			self.learn(experiences)

	def learn(self, experiences):
		self.total_it += 1

		state, action, reward, next_state, not_done = experiences
		
		done = [not ndone for ndone in not_done]
		done = torch.FloatTensor(np.vstack(np.array(done).astype(int))).to(device)
		# Sample replay buffer 

		next_action = self.actor_target(next_state)
		# Compute the target Q value
		target_Q1, target_Q2 = self.critic_target(next_state, next_action)
		target_Q = torch.min(target_Q1, target_Q2)
		target_Q = reward + done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			
		# update noise decay parameter
		if self.eps > self.eps_final:
			self.eps -= self.eps_decay
			self.eps = max(self.eps, self.eps_final)
		self.noise.reset()

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic.pth")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pth")
		torch.save(self.actor.state_dict(), filename + "_actor.pth")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pth")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pth"))
		self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth"))
