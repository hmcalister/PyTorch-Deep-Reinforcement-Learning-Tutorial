import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from itertools import count

from transitions import Transition, ReplayMemory
from DeepQNetwork import DQN
from display import plotDurations

import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("CartPole-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

numActions = env.action_space.n
state, info = env.reset()
numObservations = len(state)
policyNet = DQN(numObservations, numActions).to(device)
targetNet = DQN(numObservations, numActions).to(device)
targetNet.load_state_dict(policyNet.state_dict())

optimizer = optim.AdamW(policyNet.parameters(), lr=LR, amsgrad=True)

memory = ReplayMemory(10000)
steps = 0
episodeDurations = []


def selectAction(state):
    global steps

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)
    steps += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policyNet(state).max(1).indices.view(1, 1)
    else:
        # If we rolled below the epsilon threshold, do something random for exploration
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    


X = torch.rand(1000000, device=device)
checkGPUMemory()