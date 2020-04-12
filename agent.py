import random

import numpy as np
import torch
from pynput.keyboard import Key
from torch import nn, optim
from torch.distributions import Categorical


class Agent:
    ACTIONS = [Key.left, Key.right, Key.space]

    def action(self, state):
        return random.choice(self.ACTIONS)


class NeuralAgent(Agent, nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(NeuralAgent, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.l1 = nn.Linear(in_dim, hid_dim)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hid_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

        self.to(self.device)

    def action(self, state):
        state = state.flatten()
        state = state.astype(np.float32)
        state = torch.from_numpy(state)
        _, outputs = self.forward(state)
        action = self.sample_action(outputs)
        return self.ACTIONS[action]

    def forward(self, state):
        state = state.to(self.device)
        hidden = self.l1(state)
        hidden = self.relu(hidden)
        logits = self.l2(hidden)
        outputs = self.softmax(logits)

        return logits, outputs

    def sample_action(self, outputs):
        c = Categorical(outputs)
        action = c.sample()
        return action
