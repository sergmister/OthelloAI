from MCTS import MCTS
from NNet import NNet
from Othello.OthelloNNet.OthelloNNet import OthelloNNet


class NNetAI:

    def __init__(self, simulations=20):
        self.simulations = simulations
        self.nnet = NNet(OthelloNNet())
        self.nnet.load_checkpoint()

    def reset(self):
        self.mcts = MCTS(self.nnet, self.simulations)

    def __repr__(self):
        return NNetAI

    def move(self, state):
        probs, value, action = self.mcts.move(state)
        return action
