import numpy as np
from random import shuffle
from collections import deque
from tqdm import tqdm

from MCTS import MCTS
from NNet import NNet
from Othello.OthelloGame import OthelloGame, OthelloState
from Othello.OthelloNNet.OthelloNNet import OthelloNNet


class Coach:
    def __init__(self, game, nnet):
        self.game = game
        # How will this work with game state?\
        self.nnet = nnet
        self.simulations = 20
        # Make args container?
        self.trainExamplesHistory = deque([], maxlen=10)

    def playGame(self):
        trainExamples = []
        state = self.game.new_game()
        mcts = MCTS(self.nnet, self.simulations)
        for _ in tqdm(range(60)):
            probs, value, action = mcts.move(state)
            #print(value)
            tboard = np.full((8, 8), state.turn)
            board = np.stack((state.board, tboard))
            #trainExamples.append((board, probs, state.turn))
            trainExamples.append((board, probs, value))
            state.move(action)
            if len(state.getValidMoves()) == 0:
                won = state.getWon()
                #return [(x[0], x[1], ((-1) ** (won != x[2]))) for x in trainExamples]
                return [(x[0], x[1], x[2]) for x in trainExamples]

    def learn(self):
        for _ in range(4):
            trainExample = self.playGame()
            self.trainExamplesHistory.append(trainExample)

        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)

        self.nnet.train(trainExamples)


if __name__ == "__main__":

    nnet = NNet(OthelloNNet())
    nnet.load_checkpoint()
    coach = Coach(OthelloGame, nnet)
    for i in range(50):
        #coach.playGame()
        coach.learn()
        #print(i)
    nnet.save_checkpoint()
