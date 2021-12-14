import random

import chess
import numpy as np

from agents.AbstractAgent import AbstractAgent
from reinforcement_learning.chessnetwork import ChessNetwork

UCT_CONST = 1. / np.sqrt(2)
THRESHOLD = 0.5  # how likely is current player to win?


class Node:
    """Een top in de MCTS spelboom"""
    def __init__(self, parent, action):
        self.parent = parent  # ouder van deze top (None voor de wortel)
        self.action = action  # actie die genomen werd om hier te geraken vanuit de ouder
        self.children = []  # de kinderen van deze top
        self.explored_children = 0  # aantal kinderen die we al verkend hebben
        self.visits = 0  # aantal keer dat de top bezocht is
        self.value = 0  # lopend gemiddelde van de rollouts vanaf deze top


class MCTSAgent(AbstractAgent):
    """Schaken met MCTS"""

    def __init__(self, player, max_depth, rollouts, base_agent, uct_const=UCT_CONST):
        self.player = player  # voor welke speler krijgen wij rewards
        self.max_depth = max_depth  # maximale diepte per iteratie
        self.rollouts = rollouts  # aantal rollouts per iteratie
        self.base_agent = base_agent  # agent die we gebruiken als default policy
        self.uct_const = uct_const  # UCT constante
        self.nn = ChessNetwork()

    def uct(self, node):
        """Upper Confidence Bound for trees formule"""
        if node.visits > 0 and node.parent is not None:
            return node.value + 2 * self.uct_const * np.sqrt(2 * np.log(node.parent.visits) / node.visits)
        else:
            return np.inf

    def make_move(self, board: chess.Board) -> chess.Move:
        root = Node(None, None)
        for _ in range(self.rollouts):
            # start iteratie
            iter_board = board.copy()
            node = root

            # selectie
            while node.children and not iter_board.is_game_over():
                if node.explored_children < len(node.children):
                    child = node.children[node.explored_children]
                    node.explored_children += 1
                    node = child
                else:
                    node = max(node.children, key=self.uct)
                iter_board.push(node.action)

            # expansie
            if not iter_board.is_game_over():
                node.children = [Node(node, a) for a in iter_board.legal_moves]
                random.shuffle(node.children)

            # rollout
            node_depth = len(iter_board.move_stack)
            while not iter_board.is_game_over() and len(iter_board.move_stack) - node_depth < self.max_depth:
                # encoded_board = encode_board(iter_board, self.player)
                # probability = self.nn.model.predict(encoded_board)
                #
                # if probability >= THRESHOLD:
                #     break  # go straight to back propagation

                self.base_agent.player = iter_board.turn
                iter_board.push(self.base_agent.make_move(iter_board))
            self.base_agent.player = self.player
            reward = self.base_agent.score(iter_board)

            # negamax back-propagation
            flag = -1 if iter_board.turn == self.player else 1
            while node:
                node.visits += 1
                # update de node value met een lopend gemiddelde
                node.value += (flag * reward - node.value) / node.visits
                flag *= -1
                node = node.parent
        # Maak een zet met de huidige MCTS-boom
        node = max(root.children, key=self.uct)
        return node.action


def encode_board(board: chess.Board, player: chess.Color) -> np.array((1, 64)):
    """
    Encodes the board, so that it can be used for the network using following key: capital letters stand for white pieces
    K      0.1    k  -0.1
    Q      0.09   q  -0.09
    B      0.035  b  -0.035
    N      0.03   n  -0.03
    R      0.05   r  -0.05
    P      0.01   p  -0.01
    Empty  0

    This is from white's perspective, the values are swapped when the board is encoded from black's perspective

    :return:
    """
    arr = np.zeros((8, 8))
    flag = -1 if player else 1
    mapping = {
        None: 0,
        'K': flag * 0.1,
        'Q': flag * 0.09,
        'B': flag * 0.035,
        'N': flag * 0.03,
        'R': flag * 0.05,
        'P': flag * 0.01,
        'k': -flag * 0.1,
        'q': -flag * 0.09,
        'b': -flag * 0.035,
        'n': -flag * 0.3,
        'r': -flag * 0.5,
        'p': -flag * 0.01,
    }

    for rank in range(0, 8):
        for file in range(0, 8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            value = 0 if piece is None else mapping[piece.symbol()]

            arr[rank][file] = value

    return arr
