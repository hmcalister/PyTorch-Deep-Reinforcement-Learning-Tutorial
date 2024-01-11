import random

# namedtuple allows us to make something like a class (i.e. a set of data with associated keys) without additional boilerplate.
# deque is just a queue so we can quickly gather a lot of data and access random transitions for training. Importantly, it is fast to append!
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'nextState', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batchSize):
        """Get a random sample of observed transitions."""
        return random.sample(self.memory, batchSize)

    def __len__(self):
        return len(self.memory)