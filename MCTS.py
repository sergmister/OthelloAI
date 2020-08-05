import copy
import numpy as np


class Node:
    def __init__(self, state):
        self.state = state  # current board state
        # state turn is next person to make turn
        self.id = self.state.getID()
        self.done = 0  # check if done when we evaluate
        self.edges = set()  # other connecting nodes bellow

    def isNotLeaf(self):
        return bool(self.edges)


class Edge:
    def __init__(self, in_node, out_node, prior, turn, action):
        self.id = in_node.id + out_node.id
        self.inNode = in_node
        self.outNode = out_node
        self.turn = turn  # player who made move
        # Is this even needed?
        self.action = action  # x, y cords of move

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior,
        }

    def update_stats(self, value):
        self.stats['N'] += 1
        self.stats['W'] += value
        self.stats['Q'] = self.stats['W'] / self.stats['N']


class MCTS:
    def __init__(self, nnet, simulations=20):
        # add args?
        self.nnet = nnet
        self.tree = dict()
        self.simulations = simulations
        self.cpuct = 1
        self.EPSILON = 0.2
        self.ALPHA = 0.8

    def moveToLeaf(self):  # decides optimal Q and moves down accordingly

        breadcrumbs = []
        currentNode = self.root

        while currentNode.isNotLeaf():
            """
            Will this visit done nodes multiple times?
            """
            maxQU = -1000

            if currentNode == self.root:
                epsilon = self.EPSILON
                nu = np.random.dirichlet([self.ALPHA] * len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges)

            Nb = 0
            for edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            for idx, edge in enumerate(currentNode.edges):

                U = self.cpuct * ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * (np.sqrt(Nb) / (1 + edge.stats['N']))
                Q = edge.stats['Q']

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationEdge = edge

            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)

        return currentNode, breadcrumbs

    def evaluateLeaf(self, leaf):  # implement done variable returned?

        valid_moves = leaf.state.getValidMoves()

        if len(valid_moves) > 0:

            value, policy = self.get_preds(leaf.state.board, leaf.state.turn)

            for move in valid_moves:
                new_state = copy.deepcopy(leaf.state)
                new_state.move(move)
                if new_state.getID() not in self.tree:
                    newNode = Node(new_state)
                    self.addNode(newNode)
                else:  # making multiple paths to same node? backFill?
                    newNode = self.tree[new_state.getID()]

                newEdge = Edge(leaf, newNode, policy[move[0]][move[1]], leaf.state.turn, move)
                leaf.edges.add(newEdge)

        else:
            won = leaf.state.getWon()
            if won == leaf.state.turn:
                value = 1
            elif won == -leaf.state.turn:
                value = -1
            else:
                value = 0

        return value
        # value high if good for current player

    def backFill(self, value, breadcrumbs):
        direction = 1
        for edge in breadcrumbs:
            direction *= -1
            """
            How is MCTS supposed to deal with minimax paradoxes?
            """
            edge.update_stats(value * direction)
            if value >= 1:
                value
        """
        How do we adjust the value? - each time, times by player, etc.
        """

    def get_preds(self, state, turn):
        # Implement turn dimension
        policy, value = self.nnet.predict(state, turn)
        return value, policy

    def addNode(self, node):
        self.tree[node.id] = node

    def cleanTree(self):
        pass  # recursively remove dead branches to save memory after
        # Maybe not needed if we reset the tree after each game

    def simulate(self):
        leaf, breadcrumbs = self.moveToLeaf()
        value = self.evaluateLeaf(leaf)
        self.backFill(value, breadcrumbs)

    def move(self, state):  # if to check if root in tree?

        if state.getID() not in self.tree:
            self.addNode(Node(copy.deepcopy(state)))
        self.root = self.tree[state.getID()]

        for sim in range(self.simulations):
            self.simulate()

        value = -1000
        probs = np.zeros((8, 8))
        for node in self.root.edges:
            probs[node.action[0]][node.action[1]] = node.stats['Q']
            if node.stats['Q'] > value:
                value = node.stats['Q']
                action = node.action
        """
        Max QU? use 1 round of simulate to find move?
        # NO QU!? add prediction tho?
        Randomness for training?
        Check for game over?
        """
        return probs, value, action
