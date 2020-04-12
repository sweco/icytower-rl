import random

from pynput.keyboard import Key


class Agent:
    def action(self, state):
        return random.choice([Key.left, Key.right, Key.space])
