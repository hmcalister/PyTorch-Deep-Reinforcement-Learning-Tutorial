# All of the Neural Network stuff within PyTorch, similar to Keras for tensorflow?
import torch.nn as nn
# Use some of the neural network layers in a functional manner, for ease of network building
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, numObservations, numActions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(numObservations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, numActions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)