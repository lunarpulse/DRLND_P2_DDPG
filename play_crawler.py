from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name='./Crawler_Linux/Crawler.x86_64')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

from agent import DDPG_agent
import numpy as np
import torch

from collections import namedtuple, deque

agents = DDPG_agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=0)

agents.actor_local.load_state_dict(torch.load('crawler_checkpoint_actor.pth', map_location='cpu'))
agents.critic_local.load_state_dict(torch.load('crawler_checkpoint_critic.pth', map_location='cpu'))

env_info = env.reset(train_mode=False)[brain_name]        
states = env_info.vector_observations                  
scores = np.zeros(num_agents)                          

for i in range(1000):
    actions = agents.act(states, add_noise=False)                    
    env_info = env.step(actions)[brain_name]        
    next_states = env_info.vector_observations        
    rewards = env_info.rewards                        
    dones = env_info.local_done                 
    scores += rewards                         
    states = next_states                              
    if np.any(dones):                              
        break
print("Scores:", scores)
env.close()