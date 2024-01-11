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

    else:
        print("Device is not set to cuda")

X = torch.rand(10, device=device)
checkGPUMemory()

X = torch.rand(1000000, device=device)
checkGPUMemory()