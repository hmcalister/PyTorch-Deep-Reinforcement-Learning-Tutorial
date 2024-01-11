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
    else:

X = torch.rand(10, device=device)
checkGPUMemory()

X = torch.rand(1000000, device=device)
checkGPUMemory()