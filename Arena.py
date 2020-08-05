import time
from tqdm import tqdm
from Othello.OthelloGame import OthelloGame, OthelloState
from RandomAI import RandomAI
from NNetAI import NNetAI

class Arena:

    def __init__(self, game, ai1, ai2, rounds=100):
        self.game = game
        self.ai1 = ai1
        self.ai2 = ai2
        self.rounds = rounds

    def pit(self):
        ai1_total = 0
        ai2_total = 0
        for _ in tqdm(range(self.rounds)):
            state = self.game.new_game()
            self.ai1.reset()
            self.ai2.reset()
            while True:
                if state.turn == 1:
                    action = self.ai1.move(state)
                else:
                    action = self.ai2.move(state)
                state.move(action)
                if len(state.getValidMoves()) == 0:
                    won = state.getWon()
                    if won == 1:
                        ai1_total += 1
                    elif won == -1:
                        ai2_total += 1
                    else:
                        pass
                    break
        return ai1_total, ai2_total


if __name__ == "__main__":
    arena = Arena(OthelloGame, RandomAI(), NNetAI(), rounds=20)
    ai1, ai2 = arena.pit()
    print(ai1, ai2)
