import random


class RandomAI:

    def __init__(self):
        pass

    def reset(self):
        pass

    def __repr__(self):
        return "RandomAI"

    def move(self, state):
        action = random.choice(state.getValidMoves())
        return action
