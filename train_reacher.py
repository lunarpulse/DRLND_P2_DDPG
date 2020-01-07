import numpy as np
import torch

from collections import namedtuple, deque
import time

import pickle
import pandas as pd
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

from PPO.agent import ProximalPolicyOptimisation

def env_parse(env):
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

    return state_size, action_size, num_agents

def replay(env, agents, max_timesteps, model_ckpt):

    state_dict = torch.load(model_ckpt, map_location=lambda storage, loc: storage)
    agents.model.load_state_dict(state_dict)

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]        
    states =  to_tensor(env_info.vector_observations, device = device)                  
    scores = np.zeros(num_agents)                          

    for i in range(max_timesteps):
        actions = agents.act(states)                    
        env_info = env.step(actions.cpu().data.numpy())[brain_name]
        rewards, next_states, dones = convert(env_info, device)
        scores += rewards                         
        states = next_states                              
        if np.any(dones.cpu().data.numpy()):                              
            break
    print("Scores:", scores)

def plot(score):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(score)+1), score)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def plot_adv(fpath):
    with open(fpath, "rb") as fp:
        scores = pickle.load(fp)

    episodes = np.arange(len(scores)) + 1
    scores = pd.Series(scores)
    scores_ma = scores.rolling(100).mean()

    plt.figure(figsize=[8,4])
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.plot(episodes, scores, c="blue")
    plt.plot(episodes, scores_ma, c="red", linewidth=3)
    plt.axhline(30, linestyle="--", color="black", linewidth=2)
    plt.grid(which="major")
    plt.legend(["Episode Average Score", "Moving Average Score (100 Episodes)",
                "Criteria"])
    plt.tight_layout()
    plt.savefig("./bestmodel_score.png")

def to_tensor( x, dtype=np.float32, device = 'cpu'):
    return torch.from_numpy(np.array(x).astype(dtype)).to(device)

def convert(env_info, device):
    next_states = to_tensor(env_info.vector_observations, device = device)
    rewards = to_tensor(env_info.rewards, device = device)
    dones = to_tensor(env_info.local_done, dtype=np.uint8, device = device)
    return rewards, next_states, dones

def train(env, agents, n_episodes=2000, print_every = 10, max_t=1000, target_score = 0, model_ckpt = 'model_checkpoint.pth', device = 'cpu'):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    scores_deque = deque(maxlen=100)
    scores = []
    t0=time.time()
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = to_tensor(env_info.vector_observations, device = device)
        agents.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agents.act(states)
            env_info = env.step(actions.cpu().data.numpy())[brain_name]
            rewards, next_states, dones = convert(env_info, device)
            states = next_states
            agents.step(states, actions, rewards, next_states, dones)
            score += rewards
            if np.any(dones.cpu().data.numpy()):
                break
        scores_deque.append(score.mean())
        scores.append(score.mean())
        average_score = np.mean(scores_deque)
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.3f}'.format(i_episode, average_score, score.mean()))
            
        if average_score > target_score:
            print('\rTraining Finished at the episode {}\t Average Score: {:.2f}'.format(i_episode, average_score))
            agents.save(model_ckpt)
            break
    t1=time.time()

    print("\nTotal time elapsed: {} seconds".format(t1-t0))
    return scores


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')

    state_dim, action_dim, num_agents = env_parse(env)

    fc1 = 256
    fc2 = 128
    batch_size = 256
    agent = ProximalPolicyOptimisation(env, state_dim, action_dim, num_agents = num_agents, hiddens=[fc1, fc2],  tmax=batch_size, n_epoch=10, batch_size=batch_size, eps=0.1, device=device)

    n_episodes = 150
    print_every = 2
    max_timesteps = 2000
    target_score = 30
    model_pth = 'reacher_PPO_checkpoint.pth'

    # Training
    score_list = train(env, agent, n_episodes, print_every, max_timesteps, target_score, model_pth, device = device)

    # Plot
    plot(score_list)

    # Replay
    replay(env, agent, max_timesteps, model_pth)
    
    env.close()