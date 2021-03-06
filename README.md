[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# Continuous Control

## Introduction

The objective of this project is to train the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Distributed Training

For this project, there are two separate versions of the Unity environment:

- The first version contains a single agent.

- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### 1: First Version

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

### 2: Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores.

- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

### Prerequisites

1. Please first setup a Python3 [Anaconda](https://www.anaconda.com/download) environment.
1. Then install the requirements for the project through:

```bash
pip install -r requirement.txt
```

1. clone the repo

```bash
git clone git@github.com:ulamaca/DRLND_P2_Continuous_Control.git
```

1. Follow the instructions to download the multi-agent version environment from the [Getting Started](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control) section in Udacity DRLND repo.

1. Place the downloaded multi agent version environment to './Reacher_Linux' under the root of the repository.

## Instructions

1. Create (and activate) a new environment with Python 3.6.

- UNIX:

```bash
conda create --name drlnd python=3.6
source activate drlnd
```

1. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.

```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

1. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

1. Download the environment from one of the links below, matching your operating system:
    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

1. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies. Only use this repository to get the environment set up not using afterwards.

```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install
```

1. Place the file in the DRLND GitHub repository, in the root of this repository, and unzip (or decompress) the file.

1. Refer the notebook `Continuous_Control-DDPG.ipynb` for using Deep Deterministic Policy Gradient (DDPG).
