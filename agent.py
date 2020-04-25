import random
from collections import namedtuple

import numpy as np
import torch
from pynput.keyboard import Key
from torch import nn, optim
from torch.distributions import Categorical
from torch.nn import Conv2d, MaxPool2d, ReLU

Action = namedtuple('Action', ['action', 'log_prob'])


class Agent:
    ACTIONS_DIR = [Key.left, None, Key.right]
    ACTIONS_JUMP = [None, Key.space]

    def action(self, state):
        return random.choice(self.ACTIONS)


class NeuralAgent(Agent, nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(NeuralAgent, self).__init__()

        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

        # self.l1 = nn.Linear(in_dim, hid_dim)
        # self.relu = nn.ReLU()
        # self.l2 = nn.Linear(hid_dim, out_dim)
        self.c1 = Conv2d(2, 4, 20)
        self.p1 = MaxPool2d(20)
        self.relu = ReLU()
        self.c2 = Conv2d(4, 4, 20)
        self.p2 = nn.AdaptiveMaxPool2d(20)
        self.o1 = nn.Linear((20 ** 2) * 4, out_dim)
        self.o2 = nn.Linear((20 ** 2) * 4, 2)
        self.softmax = nn.Softmax(dim=-1)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

        self.to(self.device)

    def action(self, state):
        # state = state.flatten()
        state = np.transpose(state, [2, 0, 1])
        state = np.expand_dims(state, 0)
        state = state.astype(np.float32)
        state = torch.from_numpy(state)
        _, _, o_dir, o_jump = self.forward(state)
        a_dir, a_jump = self.sample_action(o_dir, o_jump)
        return Action(self.ACTIONS_DIR[a_dir.action], a_dir.log_prob), \
               Action(self.ACTIONS_JUMP[a_jump.action], a_jump.log_prob)

    def forward(self, state):
        state = state.to(self.device)
        # hidden = self.l1(state)
        # hidden = self.relu(hidden)
        # logits = self.l2(hidden)
        c1 = self.c1(state)
        p1 = self.p1(c1)
        p1 = self.relu(p1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        p2 = self.relu(p2)
        p2 = p2.flatten(1, -1)
        l_dir = self.o1(p2)
        l_jump = self.o2(p2)
        o_dir = self.softmax(l_dir)
        o_jump = self.softmax(l_jump)

        return l_dir, l_jump, o_dir, o_jump

    def sample_action(self, o_dir, o_jump):
        c_dir = Categorical(o_dir)
        c_jump = Categorical(o_jump)

        a_dir = c_dir.sample()
        a_jump = c_jump.sample()

        a_dir = Action(a_dir, c_dir.log_prob(a_dir))
        a_jump = Action(a_jump, c_jump.log_prob(a_jump))

        return a_dir, a_jump
