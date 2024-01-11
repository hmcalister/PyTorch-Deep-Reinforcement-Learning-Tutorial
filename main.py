import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from itertools import count

from transitions import Transition, ReplayMemory
from DeepQNetwork import DQN
from display import plotDurations

# Base PyTorch package
import torch
# All of the Neural Network stuff within PyTorch, similar to Keras for tensorflow?
import torch.nn as nn
# PyTorch built in optimization algorithms, such as ADAM
import torch.optim as optim

# Make the CartPole environment from gymnasium
env = gym.make("CartPole-v1")

# Ensure we are using a GPU if it is available.
# I was pleasantly surprised by how easy it was to set up CUDA and the GPU with pytorch!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Our goal is to train a policy (a mapping from current state to actions in that state, with some way to weight preferences of those actions)
# that maximizes a reward R_{t_0} = \sum_{t=t_0}^{\inf} \gamma^{t-t_0} r_t, i.e. the sum of discounted rewards. In effect,
# this calculation measures how much reward we expect from our *current* action, as well as how much reward we expect from the *next* action 
# based on what we did. In this way, by changing \gamma we can affect how much the policy optimizes for short-term greedy strategies compared to 
# long-term planning.
#
# Q-learning achieves this by modeling an ideal function Q*, mapping from the space of State x Action to a real number indicating the preference for that action
# (exactly as we discussed above). We then build a function Q (our model) that tries to emulate that ideal Q*. Of course, we do not *know* Q*, or
# we would not have to learn it, but we can construct an environment such that certain actions in certain states give more or less rewards to the model.
#
# Since our environment (the cart-pole) is deterministic, we can have our policy follow the Bellman equation Q^\pi(s,a) = r+\gamma Q^\pi(s", \pi(s")),
# i.e. our function Q^\pi predicts a preference for action a in state s determined by the reward r, and the Q^\pi (our preference) 
# for the MOST preferred action in the next state.
# Taking the difference of these sides gives up a temporal difference error \delta = Q^\pi(s,a) - (r+\gamma max_a (s", a)), note we replace
# our most preferred action Q^\pi(s", \pi(s") with a simple maximum over the actions in the state s".
#
# We want to minimize this temporal difference error, which we will do using the Huber loss. Other losses (e.g. MSE, MAE, etc.) would also work,
# but the Huber loss tends to smooth outliers when Q is very noisy (like at the beginning of training). Also, the tutorial is using Huber loss,
# so we will too! The Huber loss is given by L = (1/|B|) \sum_{(s, a, s", a") \in B} L(\delta), where
# L(\delta) = (1/2)\delta^2 for |\delta|<=1, and |delta|-(1/2) otherwise.

# Define some hyperparameters for the training.
#
# Note that epsilon, mentioned below, refers to the probability that we choose an action randomly (uniformly)
# rather than using the model we are learning. This helps in the exploration vs exploitation tradeoff.

# Number of transitions to sample from the replay buffer
BATCH_SIZE = 128
# Discount factor on reward. Higher numbers (close to 1) indicate a preference for long-term planning over greedy strategies
GAMMA = 0.99
# Starting  value of epsilon
EPS_START = 0.9
# Ending value of epsilon
EPS_END = 0.05
# Exponential decay rate of epsilon, with a higher number meaning slower decay
EPS_DECAY = 1000
# Update rate of the network
TAU = 0.005
# Learning rate of the ADAM optimizers
LR = 1e-4

# The number of actions possible in our environment
numActions = env.action_space.n

# The state and information of the environment as we start
state, info = env.reset()
numObservations = len(state)

# Create a policy network (to learn the policy) and a target network (to calculate our best guess at a states value without the new updates to our network)
# Put these networks on the GPU (if available) to improve training times

policyNet = DQN(numObservations, numActions).to(device)
targetNet = DQN(numObservations, numActions).to(device)
targetNet.load_state_dict(policyNet.state_dict())

# Create an optimizer (for optimizing the networks, duh)
optimizer = optim.AdamW(policyNet.parameters(), lr=LR, amsgrad=True)

# Create the memory for remembering and sampling transitions of the environment
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
    

def optimizeModel():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # This is just a big list of booleans, true if the state we sampled has a next state, false otherwise
    nonfinalMask = torch.tensor(tuple(map(lambda s: s is not None, batch.nextState)), device=device, dtype=torch.bool)
    # and this is a big list of those states that have next states
    nonfinalNextStates = torch.cat([s for s in batch.nextState if s is not None])

    stateBatch = torch.cat(batch.state)
    actionBatch = torch.cat(batch.action)
    rewardBatch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policyNet
    stateActionValues = policyNet(stateBatch).gather(1, actionBatch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for nonfinalNextStates are computed based
    # on the "older" targetNet; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    nextStateValues = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        nextStateValues[nonfinalMask] = targetNet(nonfinalNextStates).max(1).values

    # Compute the expected Q values - the targetNets estimate plus the reward of the sample
    expectedStateActionValues = (nextStateValues * GAMMA) + rewardBatch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(stateActionValues, expectedStateActionValues.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policyNet.parameters(), 100)
    optimizer.step()

# Ensure the number of episodes is reasonable for the device we are using
if torch.cuda.is_available():
    numEpisodes = 600
else:
    numEpisodes = 50

for episodeIndex in range(numEpisodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = selectAction(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            nextState = None
        else:
            nextState = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, nextState, reward)

        # Move to the next state
        state = nextState

        # Perform one step of the optimization (on the policy network)
        optimizeModel()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        targetNetStateDict = targetNet.state_dict()
        policyNetStateDict = policyNet.state_dict()
        for key in policyNetStateDict:
            targetNetStateDict[key] = policyNetStateDict[key]*TAU + targetNetStateDict[key]*(1-TAU)
        targetNet.load_state_dict(targetNetStateDict)

        if done:
            episodeDurations.append(t + 1)
            plotDurations(episodeDurations)
            break

print('Complete')
plotDurations(episodeDurations, show_result=True)
plt.show()
