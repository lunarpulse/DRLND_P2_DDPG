from unityagents import UnityEnvironment

from agent import DDPG_agent
import numpy as np
import torch

from collections import namedtuple, deque
import time

import pickle
import pandas as pd
import matplotlib.pyplot as plt

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

    return brain, brain_name, state_size, action_size, num_agents

def replay(env, max_timesteps, actor_ckpt, critic_ckpt):
    agents.actor_local.load_state_dict(torch.load(actor_ckpt, map_location='cpu'))
    agents.critic_local.load_state_dict(torch.load(critic_ckpt, map_location='cpu'))

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]        
    states = env_info.vector_observations                  
    scores = np.zeros(num_agents)                          

    for i in range(max_timesteps):
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

def train(env, brain_name, agents, n_episodes=2000, print_every = 10, max_t=1000, target_score = 0, actor_ckpt = 'model_checkpoint_actor.pth', critic_ckpt = 'nodel_checkpoint_critic.pth'):
    scores_deque = deque(maxlen=50)
    scores = []
    t0=time.time()
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
                print('\tSteps: ', t)
                break 
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))

        average_score = np.mean(scores_deque)
        if i_episode % print_every == 0 or average_score > target_score:
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.3f}'.format(i_episode, average_score, np.mean(score)))
            torch.save(agents.actor_local.state_dict(), actor_ckpt)
            torch.save(agents.critic_local.state_dict(), critic_ckpt) 
        if average_score > target_score:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
            break
    t1=time.time()

    print("\nTotal time elapsed: {} seconds".format(t1-t0))
    return scores


if __name__ == "__main__":
    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')

    brain, brain_name, state_size, action_size, num_agents = env_parse(env)

    agents = DDPG_agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=0)
    
    n_episodes = 150
    print_every = 5
    max_timesteps = 1000
    target_score = 30
    actor_ckpt = 'reacher_checkpoint_actor.pth'
    critic_ckpt = 'reacher_checkpoint_critic.pth'

    # Training
    score_list = train(env, brain_name, agents, n_episodes, print_every, max_timesteps, target_score, actor_ckpt, critic_ckpt)

    # Plot
    plot(score_list)

    # Replay
    replay(env, max_timesteps, actor_ckpt, critic_ckpt)
    
    env.close()