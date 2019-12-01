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

import time
def ddpg(n_episodes=2000, print_every = 10, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agents.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            action = agents.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agents.step(state, action, rewards, next_state, dones)
            state = next_state
            score += rewards
            if np.any(dones):
                break 
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))

        average_score = np.mean(scores_deque)
        if i_episode > 0 and i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
            torch.save(agents.actor_local.state_dict(), 'crawler_checkpoint_actor.pth')
            torch.save(agents.critic_local.state_dict(), 'crawler_checkpoint_critic.pth') 
        if average_score > 40:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
            break
    return scores

n_episodes = 10000
print_every = 50
score = ddpg(n_episodes, print_every)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(score)+1), score)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

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